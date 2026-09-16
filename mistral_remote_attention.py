from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import numpy as np

DEFAULT_MODEL = Path("/home/liuminglu/models/Mistral-7B-Instruct-v0.2")
DEFAULT_DATA = Path("/home/liuminglu/kvcache/datasets/defensivekv_dataset/longbench")
DEFAULT_OUTPUT = Path("results/gov_report_remote_attention")
SUPPORTED_TASKS = ("gov_report", "multifieldqa_en")
SCHEMA_VERSION = 1
OUTPUT_FILES = (
    "predictions.jsonl", "prefill_scores.jsonl", "checkpoint_metrics.jsonl",
    "q1_interventions.jsonl", "q2_interventions.jsonl", "journal.jsonl",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def format_prompt(tokenizer: Any, task: str, text: str) -> str:
    raw_tasks = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
    if task in raw_tasks or not getattr(tokenizer, "chat_template", None):
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def uniform_checkpoints(trajectory_length: int, count: int = 8, future_tokens: int = 8) -> list[int]:
    maximum = trajectory_length - future_tokens - 1
    if count < 1 or future_tokens < 1 or maximum < 0:
        raise ValueError("trajectory cannot supply the requested checkpoints and future targets")
    if count == 1:
        return [0]
    if maximum + 1 < count:
        raise ValueError(f"only {maximum + 1} unique checkpoints are available, need {count}")
    result = sorted({int(math.floor(i * maximum / (count - 1) + 0.5)) for i in range(count)})
    if len(result) != count:
        raise RuntimeError("uniform checkpoint rounding produced duplicates")
    return result


def stable_nll(logits: Any, target: int) -> float:
    import torch
    values = logits.detach().double().reshape(-1)
    return float(torch.logsumexp(values, 0) - values[int(target)])


def stable_kl(reference_logits: Any, altered_logits: Any) -> float:
    import torch
    p_log = torch.log_softmax(reference_logits.detach().double().reshape(-1), -1)
    q_log = torch.log_softmax(altered_logits.detach().double().reshape(-1), -1)
    return float(torch.sum(p_log.exp() * (p_log - q_log)).clamp_min(0.0))


def rankdata(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    x, y = rankdata(left), rankdata(right)
    if x.shape != y.shape or len(x) < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else None


def block_set_metrics(predicted: list[int], oracle: list[int], oracle_mass: list[float]) -> dict[str, Any]:
    p, o = set(predicted), set(oracle)
    union = p | o
    total = float(sum(oracle_mass))
    covered = float(sum(oracle_mass[i] for i in p if 0 <= i < len(oracle_mass)))
    return {
        "overlap": len(p & o),
        "recall": len(p & o) / len(o) if o else None,
        "jaccard": len(p & o) / len(union) if union else 1.0,
        "coverage": covered / total if total > 0 else None,
        "oracle_remote_mass": total,
        "predicted_covered_mass": covered,
    }


def complete_candidate_blocks(prompt_length: int, block_size: int, sink: int, local: int) -> list[dict[str, int]]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    local_start = max(sink, prompt_length - local)
    blocks = []
    for start in range(0, prompt_length - block_size + 1, block_size):
        end = start + block_size
        if start >= sink and end <= local_start:
            blocks.append({"block_idx": len(blocks), "start": start, "end": end})
    return blocks


def top_blocks(masses: list[float], count: int) -> list[int]:
    return sorted(range(len(masses)), key=lambda i: (-float(masses[i]), i))[:min(count, len(masses))]


def clone_dynamic_cache(cache: Any) -> Any:
    from transformers import DynamicCache
    pairs = []
    for layer in cache.layers:
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if keys is None or values is None:
            raise TypeError("DynamicCache layer does not expose keys and values")
        pairs.append((keys.clone(), values.clone()))
    return DynamicCache(pairs)


def cache_equal(left: Any, right: Any) -> bool:
    import torch
    return len(left.layers) == len(right.layers) and all(
        torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
        for a, b in zip(left.layers, right.layers)
    )


def cache_signature(cache: Any) -> tuple[Any, ...]:
    signature = []
    for layer_idx, layer in enumerate(cache.layers):
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if keys is None or values is None:
            raise TypeError("DynamicCache layer does not expose keys and values")
        samples = []
        for tensor in (keys, values):
            flat = tensor.detach().reshape(-1)
            indices = sorted({0, int(flat.numel()) // 2, int(flat.numel()) - 1}) if flat.numel() else []
            samples.append(tuple(float(flat[index].float().cpu()) for index in indices))
        signature.append((layer_idx, int(cache.get_seq_length(layer_idx)), tuple(keys.shape), tuple(values.shape), *samples))
    return tuple(signature)


def model_forward(model: Any, token_ids: Any, cache: Any, absolute_start: int, logits_to_keep: int = 1) -> Any:
    import torch
    length = int(token_ids.shape[1])
    positions = torch.arange(absolute_start, absolute_start + length, device=token_ids.device)
    mask = torch.ones((1, absolute_start + length), dtype=torch.long, device=token_ids.device)
    return model(
        input_ids=token_ids, attention_mask=mask, past_key_values=cache, use_cache=True,
        return_dict=True, cache_position=positions, position_ids=positions.unsqueeze(0),
        logits_to_keep=logits_to_keep,
    )


def locate_question_span(tokenizer: Any, formatted: str, context: str, question: str, raw: str) -> dict[str, Any]:
    raw_start = formatted.find(raw)
    if raw_start < 0 or formatted.find(raw, raw_start + 1) >= 0:
        raise ValueError("formatted prompt does not contain exactly one raw prompt occurrence")
    char_start = raw_start + len(context)
    char_end = char_start + len(question)
    try:
        encoded = tokenizer(formatted, add_special_tokens=True, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        indices = [i for i, (start, end) in enumerate(offsets) if end > char_start and start < char_end]
        method = "fast_tokenizer_offsets"
    except (TypeError, NotImplementedError, ValueError, KeyError):
        indices = []
        method = "fallback_last_prompt_tokens"
    reliable = bool(indices) and indices == list(range(indices[0], indices[-1] + 1))
    if reliable:
        start, end = indices[0], indices[-1] + 1
        fallback_reason = None
    else:
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=True)
        start, end = max(0, len(prompt_ids) - 64), len(prompt_ids)
        fallback_reason = "tokenizer offsets did not establish a contiguous question span"
    return {
        "start": start, "end": end,
        "query_start": max(start, end - 64), "query_end": end,
        "method": method if not reliable else "fast_tokenizer_offsets",
        "fallback": not reliable, "fallback_reason": fallback_reason,
        "char_start": char_start, "char_end": char_end,
    }


class MistralAttentionPatch:
    def __init__(self, model: Any, blocks: list[dict[str, int]], sink: int, local: int):
        self.blocks, self.sink, self.local = blocks, sink, local
        self.target_layer: int | None = None
        self.allowed_blocks: list[int] = []
        self.layer_allowed_blocks: dict[int, list[int]] | None = None
        self.capture_decode = False
        self.prefill_query: tuple[int, int] | None = None
        self.decode_masses: dict[int, dict[str, Any]] = {}
        self.prefill_masses: dict[int, dict[str, Any]] = {}
        self.prefill_head_block_masses: dict[int, list[list[float]]] = {}
        self.prompt_token_scores: dict[str, dict[int, list[list[float]]]] = {}
        self.prompt_score_configs: dict[str, tuple[int, int, str]] | None = None
        self.head_allowed_tokens: dict[int, dict[int, list[int]]] | None = None
        self.baseline_prompt_length: int | None = None
        self.originals = {}
        for layer_idx, layer in enumerate(model.model.layers):
            module = layer.self_attn
            original = module.forward
            self.originals[layer_idx] = original
            module.forward = MethodType(self._wrapper(layer_idx, original), module)

    def _allowed(self, key_length: int, query_position: int, device: Any) -> Any:
        import torch
        allowed = torch.zeros(key_length, dtype=torch.bool, device=device)
        allowed[:min(self.sink, key_length)] = True
        allowed[max(0, query_position - self.local + 1):min(query_position + 1, key_length)] = True
        for index in self.allowed_blocks:
            block = self.blocks[index]
            allowed[block["start"]:min(block["end"], key_length)] = True
        return allowed

    def _mass(self, probabilities: Any) -> dict[str, Any]:
        block_mass = [
            float(probabilities[..., b["start"]:min(b["end"], probabilities.shape[-1])].sum(-1).mean().detach())
            if b["start"] < probabilities.shape[-1] else 0.0 for b in self.blocks
        ]
        return {"block_mass": block_mass, "remote_mass": float(sum(block_mass))}

    def _wrapper(self, layer_idx: int, original: Any):
        def wrapped(module: Any, hidden_states: Any, position_embeddings: Any, attention_mask: Any,
                    past_key_values: Any = None, cache_position: Any = None, **kwargs: Any) -> Any:
            import torch
            effective = attention_mask
            cached_before = past_key_values.get_seq_length(layer_idx) if past_key_values is not None else 0
            key_length = cached_before + hidden_states.shape[1]
            masks_layer = self.target_layer == layer_idx or (
                self.layer_allowed_blocks is not None and layer_idx in self.layer_allowed_blocks
            )
            masks_heads = self.head_allowed_tokens is not None and layer_idx in self.head_allowed_tokens
            if masks_heads and hidden_states.shape[1] == 1:
                if self.baseline_prompt_length is None:
                    raise RuntimeError("baseline prompt length is required for head-specific masking")
                kv_heads = module.config.num_key_value_heads
                allowed_kv = torch.zeros((kv_heads, key_length), dtype=torch.bool, device=hidden_states.device)
                for head, indices in self.head_allowed_tokens[layer_idx].items():
                    valid = [index for index in indices if index < min(key_length, self.baseline_prompt_length)]
                    if valid:
                        allowed_kv[int(head), torch.tensor(valid, device=hidden_states.device)] = True
                allowed_kv[:, self.baseline_prompt_length:key_length] = True
                allowed_heads = allowed_kv.repeat_interleave(module.num_key_value_groups, dim=0)
                if effective is None:
                    effective = torch.zeros((1, allowed_heads.shape[0], 1, key_length), dtype=hidden_states.dtype, device=hidden_states.device)
                else:
                    effective = effective.expand(1, allowed_heads.shape[0], 1, key_length).clone()
                effective.masked_fill_(~allowed_heads[None, :, None, :], torch.finfo(effective.dtype).min)
            elif masks_layer and hidden_states.shape[1] == 1:
                old_allowed = self.allowed_blocks
                if self.layer_allowed_blocks is not None:
                    self.allowed_blocks = self.layer_allowed_blocks[layer_idx]
                allowed = self._allowed(key_length, key_length - 1, hidden_states.device)
                self.allowed_blocks = old_allowed
                if effective is None:
                    effective = torch.zeros((1, 1, 1, key_length), dtype=hidden_states.dtype, device=hidden_states.device)
                else:
                    effective = effective.clone()
                effective[..., ~allowed] = torch.finfo(effective.dtype).min
            output = original(hidden_states, position_embeddings, effective, past_key_values, cache_position, **kwargs)
            capture_prefill = self.prefill_query is not None and hidden_states.shape[1] > 1
            capture_prompt_scores = self.prompt_score_configs is not None and hidden_states.shape[1] > 1
            capture_decode = self.capture_decode and hidden_states.shape[1] == 1
            if capture_prefill or capture_decode or capture_prompt_scores:
                from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
                q = module.q_proj(hidden_states).view(1, hidden_states.shape[1], -1, module.head_dim).transpose(1, 2)
                k = past_key_values.layers[layer_idx].keys
                cos, sin = position_embeddings
                if capture_prefill:
                    start, end = self.prefill_query
                    q_prefill, _ = apply_rotary_pos_emb(q, q[:, :module.config.num_key_value_heads], cos, sin)
                    q_prefill = q_prefill[..., start:end, :]
                    keys = repeat_kv(k, module.num_key_value_groups)
                    logits = torch.matmul(q_prefill, keys.transpose(2, 3)).float() * float(module.scaling)
                    q_positions = torch.arange(start, end, device=hidden_states.device)
                    k_positions = torch.arange(keys.shape[-2], device=hidden_states.device)
                    causal = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
                    logits = logits.masked_fill(~causal[None, None, :, :], torch.finfo(logits.dtype).min)
                    probs = torch.softmax(logits, -1)
                    per_query_remote = []
                    for row, position in enumerate(range(start, end)):
                        allowed = self._allowed(keys.shape[-2], position, hidden_states.device)
                        per_query_remote.append(float(probs[..., row, ~allowed].sum(-1).mean().detach()))
                    mass = self._mass(probs)
                    mass["remote_mass"] = float(np.mean(per_query_remote))
                    mass["per_query_remote_mass"] = per_query_remote
                    self.prefill_masses[layer_idx] = mass
                    kv_heads = module.config.num_key_value_heads
                    grouped_probs = probs.view(
                        1, kv_heads, module.num_key_value_groups, probs.shape[-2], probs.shape[-1]
                    ).mean(2).mean(2)
                    self.prefill_head_block_masses[layer_idx] = [
                        [
                            float(grouped_probs[0, head, block["start"]:block["end"]].sum().detach())
                            for block in self.blocks
                        ]
                        for head in range(kv_heads)
                    ]
                    del grouped_probs
                if capture_prompt_scores:
                    import torch.nn.functional as F
                    q_all, _ = apply_rotary_pos_emb(q, q[:, :module.config.num_key_value_heads], cos, sin)
                    keys_all = repeat_kv(k, module.num_key_value_groups)
                    key_positions = torch.arange(keys_all.shape[-2], device=hidden_states.device)
                    for name, (window, kernel, pooling) in self.prompt_score_configs.items():
                        actual_window = min(window, q_all.shape[-2])
                        first_query = q_all.shape[-2] - actual_window
                        logits_window = torch.matmul(q_all[..., -actual_window:, :], keys_all.transpose(2, 3)).float() * float(module.scaling)
                        query_positions = torch.arange(first_query, q_all.shape[-2], device=hidden_states.device)
                        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                        logits_window.masked_fill_(~causal[None, None, :, :], torch.finfo(logits_window.dtype).min)
                        probabilities = torch.softmax(logits_window, -1)
                        kv_heads = module.config.num_key_value_heads
                        grouped = probabilities.view(1, kv_heads, module.num_key_value_groups, actual_window, keys_all.shape[-2]).mean(2)
                        scores = grouped.mean(2)
                        if kernel > 1:
                            pool = F.max_pool1d if pooling == "max" else F.avg_pool1d
                            scores = pool(scores, kernel_size=kernel, stride=1, padding=kernel // 2)
                        self.prompt_token_scores[name][layer_idx] = scores[0].detach().float().cpu().tolist()
                        del logits_window, probabilities, grouped, scores
                if capture_decode:
                    q, _ = apply_rotary_pos_emb(q, q[:, :module.config.num_key_value_heads], cos, sin)
                    keys = repeat_kv(k, module.num_key_value_groups)
                    logits = torch.matmul(q, keys.transpose(2, 3)).float() * float(module.scaling)
                    if attention_mask is not None:
                        logits += attention_mask[..., :keys.shape[-2]].float()
                    probs = torch.softmax(logits, -1)
                    mass = self._mass(probs)
                    old_blocks = self.allowed_blocks
                    self.allowed_blocks = []
                    allowed_a = self._allowed(keys.shape[-2], keys.shape[-2] - 1, hidden_states.device)
                    self.allowed_blocks = old_blocks
                    mass["remote_mass"] = float(probs[..., ~allowed_a].sum(-1).mean().detach())
                    self.decode_masses[layer_idx] = mass
            return output
        return wrapped

    @contextmanager
    def mode(self, target_layer: int | None = None, allowed_blocks: list[int] | None = None,
             capture_decode: bool = False, prefill_query: tuple[int, int] | None = None,
             layer_allowed_blocks: dict[int, list[int]] | None = None,
             prompt_score_configs: dict[str, tuple[int, int, str]] | None = None,
             head_allowed_tokens: dict[int, dict[int, list[int]]] | None = None,
             baseline_prompt_length: int | None = None):
        old = (self.target_layer, self.allowed_blocks, self.capture_decode, self.prefill_query,
               self.layer_allowed_blocks, self.prompt_score_configs, self.head_allowed_tokens,
               self.baseline_prompt_length)
        self.target_layer = target_layer
        self.allowed_blocks = list(allowed_blocks or [])
        self.layer_allowed_blocks = None if layer_allowed_blocks is None else {
            int(layer): list(blocks) for layer, blocks in layer_allowed_blocks.items()
        }
        self.capture_decode = capture_decode
        self.prefill_query = prefill_query
        self.prompt_score_configs = prompt_score_configs
        self.head_allowed_tokens = head_allowed_tokens
        self.baseline_prompt_length = baseline_prompt_length
        if prompt_score_configs is not None:
            self.prompt_token_scores = {name: {} for name in prompt_score_configs}
        if capture_decode:
            self.decode_masses = {}
        if prefill_query is not None:
            self.prefill_masses = {}
            self.prefill_head_block_masses = {}
        try:
            yield
        finally:
            (self.target_layer, self.allowed_blocks, self.capture_decode, self.prefill_query,
             self.layer_allowed_blocks, self.prompt_score_configs, self.head_allowed_tokens,
             self.baseline_prompt_length) = old

    def close(self, model: Any) -> None:
        for layer_idx, original in self.originals.items():
            model.model.layers[layer_idx].self_attn.forward = original


def generate_from_prefill(model: Any, prompt_output: Any, prompt_cache: Any, prompt_length: int,
                          max_new_tokens: int, eos_ids: set[int]) -> list[int]:
    import torch
    device = model.get_input_embeddings().weight.device
    output, cache = prompt_output, prompt_cache
    logits = output.logits[0, -1]
    generated = []
    for step in range(max_new_tokens):
        token_id = int(logits.argmax())
        generated.append(token_id)
        if token_id in eos_ids or step + 1 == max_new_tokens:
            break
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        del output, logits
        output = model_forward(model, token, cache, prompt_length + step)
        logits = output.logits[0, -1]
        del token
    del logits, output, cache
    return generated


def advance_cache(model: Any, cache: Any, trajectory: list[int], prompt_length: int,
                  current_checkpoint: int, target_checkpoint: int) -> int:
    import torch
    if target_checkpoint < current_checkpoint:
        raise ValueError("checkpoints must advance monotonically")
    device = model.get_input_embeddings().weight.device
    for step in range(current_checkpoint, target_checkpoint):
        token = torch.tensor([[trajectory[step]]], dtype=torch.long, device=device)
        output = model_forward(model, token, cache, prompt_length + step)
        del output, token
    return target_checkpoint


def run_branch(model: Any, patch: MistralAttentionPatch, base_cache: Any, trajectory: list[int],
               prompt_length: int, checkpoint: int, future: int, target_layer: int | None = None,
               allowed_blocks: list[int] | None = None, capture: bool = False) -> dict[str, Any]:
    cache = clone_dynamic_cache(base_cache)
    device = model.get_input_embeddings().weight.device
    logits, nll = [], []
    for step in range(future):
        import torch
        token = torch.tensor([[trajectory[checkpoint + step]]], dtype=torch.long, device=device)
        with patch.mode(
            target_layer if step == 0 else None,
            allowed_blocks if step == 0 else None,
            capture_decode=capture and step == 0,
        ):
            output = model_forward(model, token, cache, prompt_length + checkpoint + step)
        current = output.logits[0, -1].detach().cpu()
        logits.append(current)
        nll.append(stable_nll(current, trajectory[checkpoint + step + 1]))
        del output, token, current
    attention = dict(patch.decode_masses) if capture else {}
    del cache, device
    return {"logits": logits, "nll": nll, "attention": attention}


def effects(reference: dict[str, Any], altered: dict[str, Any]) -> dict[str, Any]:
    delta = [a - b for a, b in zip(altered["nll"], reference["nll"])]
    kl = [stable_kl(r, a) for r, a in zip(reference["logits"], altered["logits"])]
    return {
        "reference_nll_steps": reference["nll"], "altered_nll_steps": altered["nll"],
        "delta_nll_steps": delta, "kl_steps": kl,
        "reference_nll_mean": float(np.mean(reference["nll"])),
        "altered_nll_mean": float(np.mean(altered["nll"])),
        "delta_nll_mean": float(np.mean(delta)), "kl_mean": float(np.mean(kl)),
        "delta_nll_step0": delta[0], "kl_step0": kl[0],
    }


def recovery(a_loss: float, branch_loss: float) -> float | None:
    return (a_loss - branch_loss) / a_loss if abs(a_loss) > 1e-6 else None


def prepare_records(root: Path, tokenizer: Any, seed: int, count: int, max_context: int,
                    *, task: str = "gov_report", max_new_tokens: int = 128) -> list[dict[str, Any]]:
    from datasets import load_from_disk
    frame = load_from_disk(str(root)).to_pandas()
    subset = frame[frame["task"] == task]
    if len(subset) < count:
        raise ValueError(f"{task} has {len(subset)} rows, need {count}")
    selected = subset.sample(n=count, random_state=seed)
    records = []
    for row in selected.to_dict("records"):
        context, question = str(row["context"]), str(row["question"])
        prefix = str(row.get("answer_prefix", ""))
        raw = context + question + prefix
        formatted = format_prompt(tokenizer, task, raw)
        ids = tokenizer.encode(formatted, add_special_tokens=True)
        offset_ids = tokenizer(formatted, add_special_tokens=True, return_offsets_mapping=True)["input_ids"]
        if [int(value) for value in offset_ids] != [int(value) for value in ids]:
            raise RuntimeError(f"offset/token encoding mismatch: source_id={row['_id']}")
        if len(ids) + max_new_tokens > max_context:
            raise ValueError(
                f"prompt+max_new_tokens exceeds {max_context}: source_id={row['_id']} "
                f"prompt={len(ids)} max_new_tokens={max_new_tokens}"
            )
        span = locate_question_span(tokenizer, formatted, context, question, raw)
        records.append({
            "task": task, "source_id": str(row["_id"]), "prompt_ids": ids,
            "prompt_hash": token_hash(ids), "question_span": span,
        })
    if len(records) != count:
        raise RuntimeError("selected sample count changed unexpectedly")
    return records


def prepare_resume(output: Path, resume: bool) -> tuple[set[str], dict[str, Path]]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in OUTPUT_FILES}
    for path in paths.values():
        path.touch(exist_ok=True)
    if not resume and any(path.stat().st_size for path in paths.values()):
        raise FileExistsError("outputs exist; use --resume or choose a new output directory")
    journals = [r for r in read_jsonl(paths["journal.jsonl"]) if r.get("schema_version") == SCHEMA_VERSION]
    by_source = {str(r["source_id"]): r for r in journals if r.get("state") == "committed"}
    if len(by_source) != sum(r.get("state") == "committed" for r in journals):
        raise ValueError("duplicate committed journal source_id")
    data_rows = {
        name: [r for r in read_jsonl(path) if r.get("schema_version") == SCHEMA_VERSION]
        for name, path in paths.items() if name != "journal.jsonl"
    }
    committed = set()
    for source_id, journal in by_source.items():
        expected = journal.get("counts", {})
        valid = all(
            sum(str(row["source_id"]) == source_id for row in data_rows[name]) == int(expected.get(name, -1))
            for name in data_rows
        )
        if valid:
            committed.add(source_id)
    for name, path in paths.items():
        rows = journals if name == "journal.jsonl" else data_rows[name]
        atomic_write_jsonl(path, (r for r in rows if str(r["source_id"]) in committed))
    return committed, paths


def bootstrap(values: list[float], draws: int, seed: int) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    means = [float(np.mean([rng.choice(values) for _ in values])) for _ in range(draws)]
    return {"mean": float(np.mean(values)), "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])]}


def summarize(paths: dict[str, Path], args: Any, model: Any, seconds: float) -> dict[str, Any]:
    prefill, checkpoints = read_jsonl(paths["prefill_scores.jsonl"]), read_jsonl(paths["checkpoint_metrics.jsonl"])
    q1, q2 = read_jsonl(paths["q1_interventions.jsonl"]), read_jsonl(paths["q2_interventions.jsonl"])
    sample_ids = sorted({str(r["source_id"]) for r in prefill})
    q1_sample = []
    for source_id in sample_ids:
        ps = {int(r["layer_idx"]): float(r["prefill_remote_mass"]) for r in prefill if str(r["source_id"]) == source_id}
        layers = sorted(ps)
        decode = {layer: np.mean([float(r["decode_remote_mass"]) for r in checkpoints if str(r["source_id"]) == source_id and int(r["layer_idx"]) == layer]) for layer in layers}
        damage = {layer: np.mean([float(r["delta_nll_mean"]) for r in q1 if str(r["source_id"]) == source_id and int(r["layer_idx"]) == layer]) for layer in layers if any(str(r["source_id"]) == source_id and int(r["layer_idx"]) == layer for r in q1)}
        common = sorted(damage)
        top = min(args.important_layers, len(layers))
        ptop = set(sorted(layers, key=lambda x: (-ps[x], x))[:top])
        dtop = set(sorted(layers, key=lambda x: (-decode[x], x))[:top])
        q1_sample.append({
            "source_id": source_id, "prefill_decode_spearman": spearman([ps[x] for x in layers], [decode[x] for x in layers]),
            "prefill_damage_spearman": spearman([ps[x] for x in common], [damage[x] for x in common]),
            "top_layer_overlap": len(ptop & dtop), "top_layer_recall": len(ptop & dtop) / top if top else None,
        })
    branch_effects: dict[str, list[float]] = defaultdict(list)
    recoveries: dict[str, list[float]] = defaultdict(list)
    for row in q2:
        branch_effects[str(row["branch"])].append(float(row["delta_nll_mean"]))
        if row.get("recovery_rate") is not None:
            recoveries[str(row["branch"])].append(float(row["recovery_rate"]))
    return {
        "experiment": "mistral_prefill_decode_remote_attention", "schema_version": SCHEMA_VERSION,
        "model_path": str(args.model_path.resolve()), "model_type": model.config.model_type,
        "architecture": (model.config.architectures or [None])[0], "num_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads), "num_key_value_heads": int(model.config.num_key_value_heads),
        "task": args.task, "samples": len(sample_ids), "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size, "sink_tokens": args.sink_tokens, "local_tokens": args.local_tokens,
        "checkpoint_count": args.checkpoint_count, "future_tokens": args.future_tokens,
        "important_layers": args.important_layers, "b_blocks": args.b_blocks, "random_repeats": args.random_repeats,
        "smoke": args.smoke, "q1_per_sample": q1_sample,
        "q1_bootstrap": {
            key: bootstrap([float(r[key]) for r in q1_sample if r.get(key) is not None], args.bootstrap_draws, args.seed)
            for key in ("prefill_decode_spearman", "prefill_damage_spearman", "top_layer_recall")
        },
        "q2_block_prediction": {
            key: bootstrap(
                [float(row[key]) for row in checkpoints if row.get(key) is not None],
                args.bootstrap_draws, args.seed,
            )
            for key in ("recall", "jaccard", "coverage")
        },
        "q2_effects": {key: bootstrap(values, args.bootstrap_draws, args.seed) for key, values in branch_effects.items()},
        "q2_recovery": {key: bootstrap(values, args.bootstrap_draws, args.seed) for key, values in recoveries.items()},
        "no_op": {
            "rows": sum(r.get("branch") == "no_op" for r in q2),
            "max_abs_delta_nll": max((abs(float(r["delta_nll_mean"])) for r in q2 if r.get("branch") == "no_op"), default=None),
            "max_kl": max((float(r["kl_mean"]) for r in q2 if r.get("branch") == "no_op"), default=None),
        },
        "inference_note": "Five samples support descriptive sample-cluster bootstrap only.",
        "processing_seconds": seconds, "paths": {k: str(v.resolve()) for k, v in paths.items()},
    }


def parse_max_memory(values: list[str]) -> dict[int | str, str] | None:
    result = {}
    for value in values:
        device, limit = value.split("=", 1)
        result[int(device) if device.isdigit() else device] = limit
    return result or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test Mistral prefill remote-attention predictors and per-layer A/B interventions")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--longbench_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--task", choices=SUPPORTED_TASKS, default="gov_report")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_length", type=int, default=32768)
    parser.add_argument("--checkpoint_count", type=int, default=8)
    parser.add_argument("--future_tokens", type=int, default=8)
    parser.add_argument("--important_layers", type=int, default=8)
    parser.add_argument("--b_blocks", type=int, default=8)
    parser.add_argument("--random_repeats", type=int, default=5)
    parser.add_argument("--q1_layers", nargs="*", type=int)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--local_tokens", type=int, default=512)
    parser.add_argument("--bootstrap_draws", type=int, default=1000)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max_memory", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Force 1 sample, 2 checkpoints, 4 Q1 layers, top-2 layers, B=2, future=2, random=1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.num_samples, args.checkpoint_count, args.future_tokens = 1, 2, 2
        args.important_layers, args.b_blocks, args.random_repeats = 2, 2, 1
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if args.checkpoint_count < 1:
        raise ValueError("checkpoint_count must be positive")
    if args.future_tokens < 1:
        raise ValueError("future_tokens must be positive")
    if args.max_new_tokens < args.future_tokens + args.checkpoint_count:
        raise ValueError(
            f"max_new_tokens={args.max_new_tokens} cannot provide {args.checkpoint_count} unique "
            f"checkpoints with future_tokens={args.future_tokens}; need at least "
            f"{args.future_tokens + args.checkpoint_count}"
        )
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM
    if args.device_map == "auto" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for device_map=auto")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_path)
    expected = (config.model_type, int(config.num_hidden_layers), int(config.num_attention_heads), int(config.num_key_value_heads))
    if expected != ("mistral", 32, 32, 8):
        raise TypeError(f"expected dense Mistral 32L/32Q/8KV, got {expected}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    records = prepare_records(
        args.longbench_root, tokenizer, args.seed, args.num_samples, args.max_context_length,
        task=args.task, max_new_tokens=args.max_new_tokens,
    )
    completed, paths = prepare_resume(args.output_dir, args.resume)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": args.device_map, "low_cpu_mem_usage": True, "attn_implementation": args.attn_implementation}
    if (memory := parse_max_memory(args.max_memory)) is not None:
        kwargs["max_memory"] = memory
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
    if not isinstance(model, MistralForCausalLM) or model.config.model_type != "mistral":
        raise TypeError("loaded model must be MistralForCausalLM")
    q1_layers = list(range(32)) if args.q1_layers is None else list(dict.fromkeys(args.q1_layers))
    if args.smoke and args.q1_layers is None:
        q1_layers = list(range(4))
    if any(x < 0 or x >= 32 for x in q1_layers):
        raise ValueError("q1_layers must be in [0,32)")
    eos_ids = {int(x) for x in ([tokenizer.eos_token_id] if isinstance(tokenizer.eos_token_id, int) else [])}
    for ordinal, record in enumerate(records, 1):
        if record["source_id"] in completed:
            continue
        prompt_hash_before = token_hash(record["prompt_ids"])
        blocks = complete_candidate_blocks(len(record["prompt_ids"]), args.block_size, args.sink_tokens, args.local_tokens)
        if len(blocks) < args.b_blocks:
            raise ValueError(f"sample {record['source_id']} has only {len(blocks)} remote complete blocks")
        patch = MistralAttentionPatch(model, blocks, args.sink_tokens, args.local_tokens)
        try:
            device = model.get_input_embeddings().weight.device
            from transformers import DynamicCache
            prefill_cache = DynamicCache(config=model.config)
            prompt = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
            with torch.inference_mode(), patch.mode(prefill_query=(record["question_span"]["query_start"], record["question_span"]["query_end"])):
                prefill_output = model_forward(model, prompt, prefill_cache, 0)
                trajectory = generate_from_prefill(
                    model, prefill_output, prefill_cache, len(record["prompt_ids"]), args.max_new_tokens, eos_ids
                )
            del prefill_output, prefill_cache, prompt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            base = DynamicCache(config=model.config)
            prompt = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=device)
            with torch.inference_mode():
                base_output = model_forward(model, prompt, base, 0)
            del base_output, prompt
            base_checkpoint = 0
            try:
                checkpoints = uniform_checkpoints(
                    len(trajectory), args.checkpoint_count, args.future_tokens
                )
            except ValueError as exc:
                eos_note = " (generation ended at EOS)" if trajectory and trajectory[-1] in eos_ids else ""
                raise ValueError(
                    f"sample {record['source_id']} produced a {len(trajectory)}-token trajectory"
                    f"{eos_note}, which cannot supply checkpoint_count={args.checkpoint_count} "
                    f"and future_tokens={args.future_tokens}; increase generation length or reduce "
                    "the checkpoint/future requirements"
                ) from exc
            prefill_rows = []
            prefill_b: dict[int, list[int]] = {}
            for layer in range(32):
                mass = patch.prefill_masses[layer]
                selected = top_blocks(mass["block_mass"], args.b_blocks)
                prefill_b[layer] = selected
                prefill_rows.append({
                    "schema_version": SCHEMA_VERSION, "task": args.task, "source_id": record["source_id"],
                    "layer_idx": layer, "prompt_hash": record["prompt_hash"], "trajectory_hash": token_hash(trajectory),
                    "question_span": record["question_span"], "prefill_remote_mass": mass["remote_mass"],
                    "per_query_remote_mass": mass["per_query_remote_mass"], "block_mass": mass["block_mass"],
                    "selected_b_blocks": selected, "blocks": blocks,
                })
            important = sorted(range(32), key=lambda x: (-patch.prefill_masses[x]["remote_mass"], x))[:args.important_layers]
            prediction_rows = [{
                "schema_version": SCHEMA_VERSION, "task": args.task, "source_id": record["source_id"],
                "prompt_tokens": len(record["prompt_ids"]), "prompt_hash": record["prompt_hash"],
                "generated_token_ids": trajectory, "trajectory_hash": token_hash(trajectory),
                "prediction": tokenizer.decode(trajectory, skip_special_tokens=True), "checkpoints": checkpoints,
                "question_span": record["question_span"], "important_layers": important,
                "generation": f"FullKV greedy up to {args.max_new_tokens} tokens; downstream branches teacher-forced",
            }]
            checkpoint_rows, q1_rows, q2_rows = [], [], []
            for checkpoint in checkpoints:
                with torch.inference_mode():
                    base_checkpoint = advance_cache(
                        model, base, trajectory, len(record["prompt_ids"]), base_checkpoint, checkpoint
                    )
                base_signature = cache_signature(base)
                reference = run_branch(model, patch, base, trajectory, len(record["prompt_ids"]), checkpoint, args.future_tokens, capture=True)
                if cache_signature(base) != base_signature:
                    raise RuntimeError("reference branch polluted checkpoint cache")
                oracle_b = {}
                for layer in range(32):
                    mass = reference["attention"][layer]
                    oracle_b[layer] = top_blocks(mass["block_mass"], args.b_blocks)
                    metrics = block_set_metrics(prefill_b[layer], oracle_b[layer], mass["block_mass"])
                    checkpoint_rows.append({
                        "schema_version": SCHEMA_VERSION, "task": args.task, "source_id": record["source_id"],
                        "checkpoint": checkpoint, "layer_idx": layer, "decode_remote_mass": mass["remote_mass"],
                        "decode_block_mass": mass["block_mass"], "prefill_b_blocks": prefill_b[layer],
                        "decode_oracle_b_blocks": oracle_b[layer], **metrics,
                    })
                a_losses = {}
                for layer in q1_layers:
                    altered = run_branch(model, patch, base, trajectory, len(record["prompt_ids"]), checkpoint, args.future_tokens, layer, [])
                    result = effects(reference, altered)
                    a_losses[layer] = result["delta_nll_mean"]
                    q1_rows.append({
                        "schema_version": SCHEMA_VERSION, "task": args.task, "source_id": record["source_id"],
                        "checkpoint": checkpoint, "layer_idx": layer, "branch": "A_only",
                        "prefill_remote_mass": patch.prefill_masses[layer]["remote_mass"],
                        "decode_remote_mass": reference["attention"][layer]["remote_mass"], **result,
                    })
                rng = random.Random(args.seed + checkpoint + int(hashlib.sha256(record["source_id"].encode()).hexdigest()[:8], 16))
                for layer in important:
                    if layer not in a_losses:
                        altered_a = run_branch(model, patch, base, trajectory, len(record["prompt_ids"]), checkpoint, args.future_tokens, layer, [])
                        a_losses[layer] = effects(reference, altered_a)["delta_nll_mean"]
                    variants = [("prefill", 0, prefill_b[layer]), ("oracle", 0, oracle_b[layer])]
                    pool = list(range(len(blocks)))
                    for repeat in range(args.random_repeats):
                        variants.append(("random", repeat, sorted(rng.sample(pool, args.b_blocks))))
                    for branch, repeat, selected in variants:
                        altered = run_branch(model, patch, base, trajectory, len(record["prompt_ids"]), checkpoint, args.future_tokens, layer, selected)
                        result = effects(reference, altered)
                        q2_rows.append({
                            "schema_version": SCHEMA_VERSION, "task": args.task, "source_id": record["source_id"],
                            "checkpoint": checkpoint, "layer_idx": layer, "branch": branch, "repeat": repeat,
                            "selected_b_blocks": selected, "a_only_delta_nll_mean": a_losses[layer],
                            "a_already_sufficient": abs(a_losses[layer]) <= 1e-6,
                            "recovery_rate": recovery(a_losses[layer], result["delta_nll_mean"]), **result,
                        })
                noop = run_branch(model, patch, base, trajectory, len(record["prompt_ids"]), checkpoint, args.future_tokens, None, [])
                no_effect = effects(reference, noop)
                q2_rows.append({
                    "schema_version": SCHEMA_VERSION, "task": args.task, "source_id": record["source_id"],
                    "checkpoint": checkpoint, "layer_idx": -1, "branch": "no_op", "repeat": 0,
                    "selected_b_blocks": [], "a_only_delta_nll_mean": 0.0,
                    "a_already_sufficient": True, "recovery_rate": None, **no_effect,
                })
                if abs(no_effect["delta_nll_mean"]) > 1e-9 or no_effect["kl_mean"] > 1e-12:
                    raise RuntimeError("no-op validation failed")
                if cache_signature(base) != base_signature:
                    raise RuntimeError("intervention branch polluted checkpoint cache")
                del reference, noop, no_effect, base_signature
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if token_hash(record["prompt_ids"]) != prompt_hash_before:
                raise RuntimeError("prompt token IDs changed during processing")
            bundles = {
                "predictions.jsonl": prediction_rows, "prefill_scores.jsonl": prefill_rows,
                "checkpoint_metrics.jsonl": checkpoint_rows, "q1_interventions.jsonl": q1_rows,
                "q2_interventions.jsonl": q2_rows,
            }
            for name, rows in bundles.items():
                append_jsonl(paths[name], rows)
            append_jsonl(paths["journal.jsonl"], [{
                "schema_version": SCHEMA_VERSION, "task": args.task, "source_id": record["source_id"],
                "state": "committed", "counts": {name: len(rows) for name, rows in bundles.items()},
                "prompt_hash": record["prompt_hash"], "trajectory_hash": token_hash(trajectory),
            }])
            print(f"[{ordinal}/{len(records)}] committed {record['source_id']}", flush=True)
        finally:
            patch.close(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result = summarize(paths, args, model, time.perf_counter() - started)
    atomic_write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
