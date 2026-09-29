#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for seed in 0 1 2; do
  python experiment_scripts/train_sdf.py \
    --point_cloud_path point_cloud/armadillo.xyz \
    --experiment_name "mat_siren_s${seed}" --model_type sine \
    --lr 1e-4 --seed "$seed" --num_epochs 2000 --epochs_til_ckpt 50
  for lr in 1e-4 5e-4; do
    python experiment_scripts/train_sdf.py \
      --point_cloud_path point_cloud/armadillo.xyz \
      --experiment_name "mat_gam${lr}_s${seed}" --model_type resin \
      --hidden_features 500 --num_hidden_layers 1 --recurrent_steps 3 \
      --lr "$lr" --seed "$seed" --num_epochs 2000 --epochs_til_ckpt 50
  done
done
