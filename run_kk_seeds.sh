#!/usr/bin/env bash
set -euo pipefail

cd /root/dear-sll-vv

for seed in 42 43 44; do
  run_name="dear-p-3b-SLL-kk-downkl-seed${seed}-1k"

  echo "===== KK seed ${seed}: training starts ====="
  SEED="$seed" \
  DATA_SEED="$seed" \
  PYTHONHASHSEED="$seed" \
  RUN_NAME="$run_name" \
  bash full.sh

  echo "===== KK seed ${seed}: DL19 evaluation starts ====="
  RUN_NAME="$run_name" EVAL_SET=dl19 bash full_dl20.sh

  echo "===== KK seed ${seed}: DL20 evaluation starts ====="
  RUN_NAME="$run_name" EVAL_SET=dl20 bash full_dl20.sh

  echo "===== KK seed ${seed}: finished ====="
done
