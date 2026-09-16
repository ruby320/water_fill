#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/liuminglu/miniconda3/envs/defensivekv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/liuminglu/models/Mistral-7B-Instruct-v0.2}"
LONGBENCH_ROOT="${LONGBENCH_ROOT:-/home/liuminglu/kvcache/datasets/defensivekv_dataset/longbench}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gov_report_allocation_ablation_r20_v1}"
LOG_FILE="${LOG_FILE:-logs/gov_report_allocation_ablation_r20_v1.log}"
RESUME="${RESUME:-1}"
RANDOM_REPEATS="${RANDOM_REPEATS:-5}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_MEMORY="${MAX_MEMORY:-0=27GiB 1=27GiB 2=27GiB 3=27GiB}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_DIR}" logs
COMMAND=(
  "${PYTHON_BIN}" mistral_task_quality.py
  --model_path "${MODEL_PATH}" --longbench_root "${LONGBENCH_ROOT}"
  --output_dir "${OUTPUT_DIR}" --task gov_report
  --num_samples 5 --seed 42 --max_context_length 32768 --min_prompt_tokens 2580
  --formula_target_kv_ratios 0.20
  --methods fullkv all_a random_budget_r20 top_layer_budget_r20 formula_waterfill_r20
  --random_repeats "${RANDOM_REPEATS}"
  --block_size 128 --sink_tokens 4 --local_tokens 512 --dtype float16
  --device_map auto --attn_implementation sdpa
)
read -r -a MEMORY_LIMITS <<< "${MAX_MEMORY}"
COMMAND+=(--max_memory "${MEMORY_LIMITS[@]}")
[[ "${RESUME}" == "1" ]] && COMMAND+=(--resume)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
