#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/liuminglu/miniconda3/envs/defensivekv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/liuminglu/models/Mistral-7B-Instruct-v0.2}"
LONGBENCH_ROOT="${LONGBENCH_ROOT:-/home/liuminglu/kvcache/datasets/defensivekv_dataset/longbench}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_MEMORY="${MAX_MEMORY:-0=27GiB 1=27GiB 2=27GiB 3=27GiB}"
RESUME="${RESUME:-1}"
RUN_TAG="${RUN_TAG:-waterfill_sweep_v1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
read -r -a MEMORY_LIMITS <<< "${MAX_MEMORY}"

for TASK in gov_report hotpotqa lcc multi_news; do
  OUTPUT_DIR="results/${TASK}_task_quality_${RUN_TAG}"
  LOG_FILE="logs/${TASK}_task_quality_${RUN_TAG}.log"
  mkdir -p "${OUTPUT_DIR}" logs
  COMMAND=(
    "${PYTHON_BIN}" mistral_task_quality.py
    --task "${TASK}" --model_path "${MODEL_PATH}" --longbench_root "${LONGBENCH_ROOT}"
    --output_dir "${OUTPUT_DIR}" --num_samples 5 --seed 42 --max_context_length 32768
    --min_prompt_tokens 2580
    --important_layers 8 --budgets 4 8 12 16 --random_repeats 5
    --block_size 128 --sink_tokens 4 --local_tokens 512
    --formula_target_kv_ratio 0.20 --formula_target_kv_ratios 0.18 0.20 0.25 0.30 --dtype float16
    --device_map auto --attn_implementation sdpa --max_memory "${MEMORY_LIMITS[@]}"
  )
  [[ "${TASK}" == "hotpotqa" ]] && COMMAND+=(--require_answer_evidence)
  [[ "${RESUME}" == "1" ]] && COMMAND+=(--resume)
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
done
