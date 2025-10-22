#!/usr/bin/env bash
# set -x
export PYTHONPATH=$(pwd)

ROOT_DIR=$(pwd)

DATA_DIR="${ROOT_DIR}/data/CROHME"
DATA_OUT_DIR="${ROOT_DIR}/data/processed"
TRAIN_DIR="${ROOT_DIR}/data/processed/train.jsonl"


python datamodule/process_data.py \
    --data_dir "${DATA_DIR}" \
    --out_dir "${DATA_OUT_DIR}" \
    --val_ratio 0.12 \
    --seed 5506

