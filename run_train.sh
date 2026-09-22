#!/usr/bin/env bash
set -euo pipefail

####
source xxx
conda activate jev

cd xxx

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 兼容两种训练机器的模型挂载路径（优先使用带 -main 的目录）。
python - <<'PYMODEL'
from pathlib import Path

preferred = Path('xxx')
fallback = Path('xxx')
source = preferred if preferred.is_dir() else fallback
if not source.is_dir():
    raise SystemExit(f'Model directory not found: {preferred} or {fallback}')

checkpoint = Path('checkpoint/Qwen3-0.6B')
checkpoint.parent.mkdir(parents=True, exist_ok=True)
if checkpoint.is_symlink():
    checkpoint.unlink()
    checkpoint.symlink_to(source, target_is_directory=True)
elif checkpoint.exists():
    if not checkpoint.is_dir():
        raise SystemExit(f'Model checkpoint path is not a directory: {checkpoint}')
    # 兼容已有的文件级软链接，不覆盖真实文件。
    entries = list(source.iterdir())
    for entry in entries:
        link = checkpoint / entry.name
        if link.exists() and not link.is_symlink():
            raise SystemExit(f'Refusing to overwrite a real checkpoint file: {link}')
    for entry in entries:
        link = checkpoint / entry.name
        if link.is_symlink():
            link.unlink()
        link.symlink_to(entry, target_is_directory=entry.is_dir())
else:
    checkpoint.symlink_to(source, target_is_directory=True)
print(f'Model path: {checkpoint} -> {source}', flush=True)
PYMODEL

CONFIG="${1:-config/train.yaml}"
# 每次启动生成独立输出目录；所有 DDP 进程共用同一路径。
OUT="$(python - "$CONFIG" <<'PYOUT'
import sys
from datetime import datetime
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = str(config["out"]).rstrip("/")
print(f"{base}_{datetime.now():%Y_%m_%d_%H%M%S_%f}")
PYOUT
)"
printf 'Training output: %s\n' "$OUT"

# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}" # 多卡示例：0,1,2,3

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" # 多卡示例：0,1,2,3

IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${#GPUS[@]}"

exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    -m jev_mini.train --config "$CONFIG" --out "$OUT"




