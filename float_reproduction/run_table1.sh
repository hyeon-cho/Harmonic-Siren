#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gpu="${1:-0}"
for omega in 32 64 128 256; do
  python run.py --cfg "cfgs/General/truth_omega${omega}.yaml" --gpu "$gpu" \
    --seed 0 --save-root save --name "table1_omega${omega}" -b
done
