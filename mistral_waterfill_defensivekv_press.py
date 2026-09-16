from __future__ import annotations

from typing import Any

import torch

from mistral_defensivekv_press import MistralDefensiveKVPress


class ExactLayerBudgetMistralDefensiveKVPress(MistralDefensiveKVPress):
    """Run the DefensiveKV scorer with an exact flattened KV-element target per layer."""

    def __init__(self, layer_targets: dict[int, int]):
        super().__init__(compression_ratio=0.0)
        self.layer_targets = {int(layer): int(target) for layer, target in layer_targets.items()}
        self.actual_kept: dict[int, int] = {}

    def compress(self, module: Any, hidden_states: torch.Tensor, keys: torch.Tensor,
                 values: torch.Tensor, attentions: torch.Tensor,
                 kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        cache = kwargs.get("past_key_value", kwargs.get("past_key_values"))
        if cache is None:
            raise ValueError("DefensiveKV compression requires a cache")
        layer = int(module.layer_idx)
        if layer not in self.layer_targets:
            raise KeyError(f"missing exact target for layer {layer}")
        metadata = cache.metadata_list[layer]
        batch_size = int(hidden_states.shape[0])
        if batch_size != 1:
            raise ValueError("exact DefensiveKV compression supports batch size 1 only")
        total_elements = int(keys.shape[0])
        n_kept = self.layer_targets[layer]
        if not 0 < n_kept <= total_elements:
            raise ValueError(f"layer {layer} target {n_kept} is outside [1, {total_elements}]")

        kwargs["metadata"] = metadata
        previous_ratio = self.compression_ratio
        self.compression_ratio = 1.0 - n_kept / total_elements
        try:
            with torch.no_grad():
                flatten_scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        finally:
            self.compression_ratio = previous_ratio
        if flatten_scores.shape != (batch_size, total_elements):
            raise RuntimeError(f"unexpected score shape {tuple(flatten_scores.shape)} for {total_elements} elements")

        topk_indices = flatten_scores.topk(n_kept, dim=-1).indices
        prompt_length = int(metadata.head_lens[0].item())
        head_indices = topk_indices // prompt_length
        num_heads = int(metadata.num_key_value_heads)
        compressed_head_lens = torch.zeros((batch_size, num_heads), dtype=torch.int32, device=keys.device)
        for batch in range(batch_size):
            compressed_head_lens[batch].scatter_add_(
                0, head_indices[batch], torch.ones_like(head_indices[batch], dtype=torch.int32)
            )
        cumulative = torch.cumsum(compressed_head_lens, dim=1, dtype=torch.int32)
        offsets = torch.arange(0, n_kept * batch_size, n_kept, device=keys.device,
                               dtype=torch.int32).view(-1, 1)
        cu_seqlens = torch.cat((torch.zeros(1, dtype=torch.int32, device=keys.device),
                               (cumulative + offsets).reshape(-1)))
        flat_head_lens = compressed_head_lens.reshape(-1)
        metadata._update_metadata_while_compressing(
            flat_head_lens, cu_seqlens, int(flat_head_lens.max().item())
        )

        order = torch.argsort(head_indices, dim=-1, stable=True)
        sorted_indices = torch.gather(topk_indices, 1, order)
        batch_offsets = torch.arange(batch_size, device=keys.device,
                                     dtype=sorted_indices.dtype).view(-1, 1) * total_elements
        gather_indices = (sorted_indices + batch_offsets).reshape(-1, 1).expand(-1, int(module.head_dim))
        compressed_keys = keys.gather(0, gather_indices).contiguous()
        compressed_values = values.gather(0, gather_indices).contiguous()
        actual = int(compressed_keys.shape[0])
        if actual != n_kept or int(flat_head_lens.sum().item()) != n_kept:
            raise RuntimeError(f"layer {layer} retained {actual}, expected {n_kept}")
        self.actual_kept[layer] = actual
        return compressed_keys, compressed_values
