#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/liuminglu/miniconda3/envs/defensivekv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/liuminglu/models/Mistral-7B-Instruct-v0.2}"
LONGBENCH_ROOT="${LONGBENCH_ROOT:-/home/liuminglu/kvcache/datasets/defensivekv_dataset/longbench}"
SOURCE_PREDICTIONS="${SOURCE_PREDICTIONS:-results/gov_report_remote_attention/predictions.jsonl}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_MEMORY="${MAX_MEMORY:-0=27GiB 1=27GiB 2=27GiB 3=27GiB}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-results/gov_report_b_budget_sweep_smoke}"
  LOG_FILE="${LOG_FILE:-logs/gov_report_b_budget_sweep_smoke.log}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-results/gov_report_b_budget_sweep}"
  LOG_FILE="${LOG_FILE:-logs/gov_report_b_budget_sweep.log}"
fi
mkdir -p "${OUTPUT_DIR}" logs
COMMAND=(
  "${PYTHON_BIN}" mistral_b_budget_sweep.py
  --model_path "${MODEL_PATH}" --longbench_root "${LONGBENCH_ROOT}"
  --source_predictions "${SOURCE_PREDICTIONS}" --output_dir "${OUTPUT_DIR}"
  --num_samples 5 --seed 42 --max_context_length 32768
  --checkpoint_count 8 --future_tokens 8 --important_layers 8
  --budgets 4 8 12 16 24 --random_repeats 5 --block_size 128
  --sink_tokens 4 --local_tokens 512 --dtype float16
  --device_map auto --attn_implementation sdpa
)
read -r -a MEMORY_LIMITS <<< "${MAX_MEMORY}"
COMMAND+=(--max_memory "${MEMORY_LIMITS[@]}")
[[ "${RESUME}" == "1" ]] && COMMAND+=(--resume)
[[ "${SMOKE}" == "1" ]] && COMMAND+=(--smoke)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
