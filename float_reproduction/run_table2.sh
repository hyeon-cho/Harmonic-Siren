#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gpu="${1:-0}"
for step in 1 2 3 4 5; do
  python run.py --cfg "cfgs/General/truth_R${step}.yaml" --gpu "$gpu" \
    --seed 0 --save-root save --name "table2_R${step}" -b
done
