#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuminglu/miniconda3/envs/defensivekv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/liuminglu/models/Mistral-7B-Instruct-v0.2}"
LONGBENCH_ROOT="${LONGBENCH_ROOT:-/home/liuminglu/kvcache/datasets/defensivekv_dataset/longbench}"
TASK="${TASK:-multifieldqa_en}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"; SEED="${SEED:-42}"; MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-32768}"
CHECKPOINT_COUNT="${CHECKPOINT_COUNT:-8}"; FUTURE_TOKENS="${FUTURE_TOKENS:-8}"
IMPORTANT_LAYERS="${IMPORTANT_LAYERS:-8}"; B_BLOCKS="${B_BLOCKS:-8}"; RANDOM_REPEATS="${RANDOM_REPEATS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"; SINK_TOKENS="${SINK_TOKENS:-4}"; LOCAL_TOKENS="${LOCAL_TOKENS:-512}"
DTYPE="${DTYPE:-float16}"; DEVICE_MAP="${DEVICE_MAP:-auto}"; ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SMOKE="${SMOKE:-0}"; RESUME="${RESUME:-1}"; CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MAX_MEMORY="${MAX_MEMORY:-0=27GiB 1=27GiB 2=27GiB 3=27GiB}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-results/multifieldqa_en_remote_attention_smoke}"
  LOG_FILE="${LOG_FILE:-logs/multifieldqa_en_remote_attention_smoke.log}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-results/multifieldqa_en_remote_attention}"
  LOG_FILE="${LOG_FILE:-logs/multifieldqa_en_remote_attention.log}"
fi
mkdir -p "${OUTPUT_DIR}" logs
COMMAND=("${PYTHON_BIN}" mistral_remote_attention.py
  --model_path "${MODEL_PATH}" --longbench_root "${LONGBENCH_ROOT}" --task "${TASK}"
  --max_new_tokens "${MAX_NEW_TOKENS}" --output_dir "${OUTPUT_DIR}"
  --num_samples "${NUM_SAMPLES}" --seed "${SEED}" --max_context_length "${MAX_CONTEXT_LENGTH}"
  --checkpoint_count "${CHECKPOINT_COUNT}" --future_tokens "${FUTURE_TOKENS}"
  --important_layers "${IMPORTANT_LAYERS}" --b_blocks "${B_BLOCKS}" --random_repeats "${RANDOM_REPEATS}"
  --block_size "${BLOCK_SIZE}" --sink_tokens "${SINK_TOKENS}" --local_tokens "${LOCAL_TOKENS}"
  --dtype "${DTYPE}" --device_map "${DEVICE_MAP}" --attn_implementation "${ATTN_IMPLEMENTATION}")
read -r -a MEMORY_LIMITS <<< "${MAX_MEMORY}"
COMMAND+=(--max_memory "${MEMORY_LIMITS[@]}")
[[ "${RESUME}" == "1" ]] && COMMAND+=(--resume)
[[ "${SMOKE}" == "1" ]] && COMMAND+=(--smoke)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
