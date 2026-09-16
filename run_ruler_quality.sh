#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuminglu/miniconda3/envs/defensivekv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/liuminglu/models/Mistral-7B-Instruct-v0.2}"
RULER_ROOT="${RULER_ROOT:-/home/liuminglu/kvcache/datasets/defensivekv_dataset/ruler}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_MEMORY="${MAX_MEMORY:-0=27GiB 1=27GiB 2=27GiB 3=27GiB}"
OUTPUT_DIR="${OUTPUT_DIR:-results/ruler_16k_all_methods_r20_v1}"
LOG_FILE="${LOG_FILE:-logs/ruler_16k_all_methods_r20_v1.log}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
read -r -a MEMORY_LIMITS <<< "${MAX_MEMORY}"
mkdir -p "${OUTPUT_DIR}" logs
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" mistral_ruler_quality.py \
  --model_path "${MODEL_PATH}" --ruler_root "${RULER_ROOT}" --output_dir "${OUTPUT_DIR}" \
  --context_length 16384 --tasks qa_1 niah_multivalue niah_multikey_3 cwe \
  --num_samples 5 --seed 42 --target_kv_ratio 0.20 \
  --block_size 128 --sink_tokens 4 --local_tokens 512 --dtype float16 \
  --device_map auto --attn_implementation sdpa --max_memory "${MEMORY_LIMITS[@]}" \
  --resume "$@" 2>&1 | tee "${LOG_FILE}"
