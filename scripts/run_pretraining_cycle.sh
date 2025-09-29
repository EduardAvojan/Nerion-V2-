#!/usr/bin/env bash
set -euo pipefail

DATASET_DB=${1:-out/learning/curriculum.sqlite}
DATASET_ROOT=${2:-experiments/datasets/gnn/latest}
PRETRAIN_OUT=${3:-experiments/runs/gnn_pretrain}
TRAIN_OUT=${4:-experiments/runs/gnn}
LR=${5:-5e-4}
EPOCHS=${6:-25}
VAL_RATIO=${7:-0.2}

build_dataset() {
  python -m nerion_digital_physicist.training.dataset_builder \
    --db "$DATASET_DB" \
    --output-dir "$DATASET_ROOT" \
    --mode pretrain
}

run_pretrain() {
  latest_dir=$(ls -td "$DATASET_ROOT"/pretrain/* | head -n1)
  python -m nerion_digital_physicist.training.pretrain \
    --dataset "$latest_dir/dataset.pt" \
    --output-dir "$PRETRAIN_OUT"
  echo "$latest_dir/dataset.pt"
}

run_finetune() {
  dataset_path="$DATASET_ROOT"/$(ls -t "$DATASET_ROOT" | grep -v pretrain | head -n1)/dataset.pt
  python -m nerion_digital_physicist.training.run_training \
    --dataset "$dataset_path" \
    --output-dir "$TRAIN_OUT" \
    --architecture gat \
    --hidden-channels 256 \
    --pooling sum \
    --residual \
    --attention-heads 4 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --dropout 0.3 \
    --val-ratio "$VAL_RATIO" \
    --pretrained digital_physicist_pretrain.pt
}

build_dataset
pre_dataset=$(run_pretrain)
echo "Pretraining dataset: $pre_dataset"
run_finetune
