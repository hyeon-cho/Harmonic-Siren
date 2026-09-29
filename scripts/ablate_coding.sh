#!/bin/bash
# Table 9: Gray coding (GC) vs. natural binary coding (NBC).
# Sets the bit-coding for BOTH the train and valid wrappers; the trainer picks up
# the same key for inference recomposition (see trainers/base_trainer.py).
# Usage: bash scripts/ablate_coding.sh [GPU] [CFG]
set -e
GPU=${1:-0}
CFG=${2:-cfgs/General/ReSIN.yaml}

for CODING in gray binary; do
    echo "=== coding = ${CODING} ==="
    python run.py --cfg "$CFG" --gpu "$GPU" \
        --opt datasets.train.wrapper.args.transformer "$CODING" \
              datasets.valid.wrapper.args.transformer "$CODING" \
        --tag "${CODING}"
done
