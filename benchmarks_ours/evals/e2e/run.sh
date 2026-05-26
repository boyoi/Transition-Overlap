#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_FLASHINFER="${VLLM_USE_FLASHINFER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Use all visible GPUs by default (vLLM tensor parallel).
if [[ -z "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
  IFS=',' read -r -a _vllm_gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
  export VLLM_TENSOR_PARALLEL_SIZE="${#_vllm_gpus[@]}"
fi

model="${MODEL:-Qwen/Qwen3-4B}"

dataset="${DATASET:-2wikimqa}"
approach="${APPROACH:-SSO}"

cd "${SCRIPT_DIR}"

python3 main.py \
      --model "${model}" \
      --dataset "${dataset}" \
      --approach "${approach}"
