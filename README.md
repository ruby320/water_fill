# Mistral prefill/decode remote-attention experiment

This directory is independent of `motivation2` and `motivation3`. It tests two descriptive questions on five seed-42 LongBench `gov_report` examples using native FullKV greedy generation from `Mistral-7B-Instruct-v0.2`.

## Definitions

- Prompt: `context + question + answer_prefix`, then the existing `format_prompt` chat-template policy. No truncation is permitted. The run fails if `prompt_tokens + 128 > 32768`.
- Prefill query: the final at most 64 tokens of the exact question span. The fast tokenizer's character offsets establish the exact span when available. If they cannot establish a contiguous span, the experiment explicitly records `fallback=true`, its reason, and uses the final at most 64 prompt tokens; it never silently claims those tokens are the question span.
- A at absolute query position `q`: sink `[0,4)` plus causal local `[q-511,q]`. Each prefill query therefore has its own causal A.
- Candidate blocks: complete 128-token blocks on the prompt's fixed start-aligned grid that are wholly outside decode-time A. Partial boundary blocks are excluded.
- Prefill score: mean remote attention mass across question queries and heads. Attention is recomputed only for the selected queries against cached K; no layer-by-layer `L x L` attention tensor is requested or stored.
- Important layers: per-sample top 8 layers by continuous prefill remote mass.
- B: independent per-layer top 8 candidate blocks by full block attention mass.
- Decode checkpoint `c`: cache contains prompt plus generated IDs `y[:c]`; the branch submits `y[c]` and scores `y[c+1]`. Eight checkpoints are uniform over positions with eight complete future targets.
- Intervention: only the target layer's current query is restricted to A or A union B. Subsequent teacher-forced steps are unmasked, so later effects propagate only through the altered current-token K/V state.

Q1 runs per-layer A-only interventions (all 32 layers formally). Q2 runs A+B-prefill, A+B-decode-oracle, and five independently sampled random-B controls only for the eight important layers; all other layers remain FullKV. NLL and forward KL are reported per step, without a synthetic mixed score. Recovery is `(A-only delta NLL - branch delta NLL) / A-only delta NLL`, and both raw losses are retained.

## Outputs and resume

Each sample is buffered in memory, its rows are fsynced to all output files, and a `journal.jsonl` commit record is written last. `--resume` keeps only rows belonging to committed samples, removing partial tails before continuing.

- `predictions.jsonl`: prompt/generation hashes, fixed token IDs, decoded prediction, checkpoints, question-span provenance, important layers.
- `prefill_scores.jsonl`: continuous layer score, every block mass, per-query remote mass, and B selection.
- `checkpoint_metrics.jsonl`: decode continuous layer score, every block mass, oracle selection, overlap/recall/Jaccard/coverage.
- `q1_interventions.jsonl`: all A-only per-step and mean NLL/KL effects.
- `q2_interventions.jsonl`: prefill/oracle/random branches, raw A-only loss, branch loss, per-step metrics, and recovery.
- `journal.jsonl`: transaction commits and row counts.
- `summary.json`: per-sample Q1 correlations/top-layer overlap and descriptive sample bootstrap; Q2 branch effects/recovery and no-op bounds.

## Commands

```bash
python -m py_compile mistral_remote_attention.py test_mistral_remote_attention.py
python -m unittest -v test_mistral_remote_attention.py
bash -n run_gov_report_remote_attention.sh
python mistral_remote_attention.py --help
SMOKE=1 RESUME=0 OUTPUT_DIR=results/smoke ./run_gov_report_remote_attention.sh
./run_gov_report_remote_attention.sh
```

Smoke mode is hard-limited to one sample, two checkpoints, Q1 layers 0-3, top two important layers, B=2, future=2, and one random repeat. Formal defaults remain 5/8/all-32/top-8/B8/future-8/random-5.
## Independent B-budget sweep

`mistral_b_budget_sweep.py` reuses the attention/cache helpers above but writes only to `results/gov_report_b_budget_sweep`. It strictly joins the current seed-42 records to `results/gov_report_remote_attention/predictions.jsonl`, validates prompt and trajectory hashes, and uses the stored generated token IDs, checkpoints, and important layers. It never regenerates a trajectory.

The sweep tests requested B budgets 4/8/12/16/24 for prefill-ranked, decode-attention-oracle, and five nested random controls. The effective budget is `min(requested_budget, candidate_count)`; saturated rows retain both values. In particular, the 3033-token sample has 18 complete candidate blocks, so B24 is reported as effective B18. `summary.json` includes both all-sample and unsaturated-sample-only reports.

Each sample performs one captured prefill, advances one base cache monotonically, and evaluates one FullKV reference and one A-only branch per important layer/checkpoint. The primary restoration metric is `(KL_A - KL_branch) / KL_A`; cells with `KL_A <= 1e-12` are marked `A_already_sufficient` and excluded from that ratio. Random repeats are averaged within a cell before layer/checkpoint/sample aggregation, and bootstrap intervals resample the five samples.

Outputs are `metadata.jsonl`, `block_metrics.jsonl`, `a_only.jsonl`, `interventions.jsonl`, `journal.jsonl`, and `summary.json`. Resume commits transactionally by sample.

```bash
python -m py_compile mistral_b_budget_sweep.py test_mistral_b_budget_sweep.py
python -m unittest -v test_mistral_b_budget_sweep.py
bash -n run_b_budget_sweep.sh
python mistral_b_budget_sweep.py --help
SMOKE=1 RESUME=0 OUTPUT_DIR=results/gov_report_b_budget_sweep_smoke ./run_b_budget_sweep.sh
./run_b_budget_sweep.sh
```

Smoke mode is hard-limited to one sample, one stored checkpoint, two stored important layers, future=2, budgets 4/8, and one random repeat. Formal defaults are five samples, eight stored checkpoints, eight stored important layers, future=8, budgets 4/8/12/16/24, and five random repeats.


## MultiFieldQA mechanism run and plotting

Run the MultiFieldQA mechanism experiment before plotting:

```bash
./run_multifieldqa_remote_attention.sh
```

Then plot GovReport and MultiFieldQA with the same mechanism definitions:

```bash
python plot_ab_remote_evidence.py \
  --input-dir results/gov_report_remote_attention \
  --input-dir results/multifieldqa_en_remote_attention \
  --output-dir results/ab_remote_evidence
```

The second `--input-dir` is optional; omit it to plot GovReport only. MultiFieldQA must first be run through the mechanism experiment above. Existing task-quality JSON files do not contain block-level attention or intervention KL and therefore cannot produce this mechanism figure.

## Equal-total-budget allocation ablation

`mistral_task_quality.py` compares B allocations under the exact same per-sample budget from `formula_total_budget`. The figure-facing random baseline is `random_budget_r20`: for each sample and repeat, it uniformly samples the exact total budget without replacement from all `num_layers * candidate_count` feasible `(layer_idx, remote_block_idx)` pairs. Both the layer allocation and within-layer block positions are therefore random; it does not first divide the budget evenly among layers. Its stable SHA-256-derived seed depends on the run seed, source ID, and repeat, and each row records the repeat, selected blocks, random seed, layer counts, exact target/allocated budget, achieved ratio, and allocation rule. Repeats are averaged within each sample before the cross-sample summary. The default is five repeats through `--random_repeats`.

`uniform_budget_r20` remains available as an optional baseline and divides the budget across all 32 layers as evenly as possible, assigning remainder blocks by ascending layer index. `top_layer_budget_r20` is a strict-budget ranked-fill baseline: it ranks all 32 layers by descending prefill remote mass (layer index breaks ties), fills each ranked layer to `candidate_count`, and partially fills the final active layer. Thus it preserves the Top-layer B preference without dropping budget when the nominal top eight layers lack capacity. Within every layer, these deterministic baselines retain the highest-mass blocks.

Run the five-sample GovReport allocation ablation on four visible GPUs (task-default generation length, 512 tokens):

```bash
bash -n run_allocation_ablation.sh
./run_allocation_ablation.sh
```

The recommended figure methods are `fullkv all_a random_budget_r20 top_layer_budget_r20 formula_waterfill_r20`. Resume keys include `repeat`, so an existing output directory containing the other methods adds all five missing random rows per sample rather than collapsing them to one. Override `CUDA_VISIBLE_DEVICES`, `MAX_MEMORY`, `OUTPUT_DIR`, `LOG_FILE`, `RESUME`, or `RANDOM_REPEATS` through environment variables when needed.
