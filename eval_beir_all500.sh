#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/root/dear-sll-vv
OUTPUT_ROOT=/hy-tmp/DeAR/outputs
RESULT_BASE=/root/dear_mse/results/qk-mvp-all500-beir8

mkdir -p "$RESULT_BASE"

for seed in 42 43 44; do
  ckpt_root="$OUTPUT_ROOT/dear-qk-mvp-5pass-seed${seed}-earlystop-new"
  for ckpt in "$ckpt_root"/adapter-step-*; do
    [[ -d "$ckpt" ]] || continue
    step_name=$(basename "$ckpt")
    result_dir="$RESULT_BASE/seed${seed}-${step_name}"

    if [[ -s "$result_dir/beir8_summary.tsv" && -s "$result_dir/beir8_mean.log" ]]; then
      echo "[$(date '+%F %T')] skip completed seed=${seed} ${step_name}"
      continue
    fi

    echo "[$(date '+%F %T')] evaluating seed=${seed} checkpoint=${step_name}"
    CKPT_DIR="$ckpt" \
    RESULT_ROOT="$result_dir" \
    CODE_DIR="$CODE_DIR" \
      bash "$CODE_DIR/full_beir.sh"
  done
done

echo "[$(date '+%F %T')] all 500-step checkpoint BEIR evaluations completed"
