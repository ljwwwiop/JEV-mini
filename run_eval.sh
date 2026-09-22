#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

CONFIG="${1:-config/eval.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python -m jev_mini.evaluate --config "$CONFIG"
