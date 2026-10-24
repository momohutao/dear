#!/usr/bin/env bash
set -euo pipefail

cd /root/dear-sll-vv

for seed in 43 44; do
  run_name="dear-p-3b-SLL-qk-downkl-seed${seed}-1k"

  echo "===== QK seed ${seed}: training starts ====="
  SEED="$seed" \
  DATA_SEED="$seed" \
  PYTHONHASHSEED="$seed" \
  RUN_NAME="$run_name" \
  LAMBDA_RAT=0.02 \
  LAMBDA_KK=0 \
  bash full.sh

  echo "===== QK seed ${seed}: DL20 evaluation starts ====="
  RUN_NAME="$run_name" EVAL_SET=dl20 bash full_dl20.sh

  echo "===== QK seed ${seed}: finished ====="
done
