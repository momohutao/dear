#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=/root/dear-sll-vv
OUTPUT_ROOT=/hy-tmp/DeAR/outputs
RESULT_BASE=/root/dear_mse/results/qk-mvp-best-beir8
SEED44_TRAIN_PID="${SEED44_TRAIN_PID:-519135}"

mkdir -p "$RESULT_BASE"

if [[ -n "$SEED44_TRAIN_PID" && "$SEED44_TRAIN_PID" != "0" ]]; then
  echo "[$(date '+%F %T')] Waiting for seed44 training job PID $SEED44_TRAIN_PID"
  while kill -0 "$SEED44_TRAIN_PID" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date '+%F %T')] Seed44 training job has exited; starting BEIR-8 evaluation"
else
  echo "[$(date '+%F %T')] No training PID supplied; starting BEIR-8 evaluation"
fi

SEEDS="${SEEDS:-42 43 44}"
for seed in $SEEDS; do
  state_file="$OUTPUT_ROOT/dear-qk-mvp-5pass-seed${seed}-earlystop-new/early_stop_state.json"

  if [[ ! -s "$state_file" ]]; then
    echo "ERROR: missing early-stop state: $state_file" >&2
    exit 1
  fi

  best_checkpoint=$(
    python - "$state_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    state = json.load(handle)

history = state.get("history", [])
if not history:
    raise SystemExit("empty early-stop history")

best = max(history, key=lambda item: item["mean_ndcg_cut_10"])
print(best["best_checkpoint"])
PY
  )

  if [[ ! -s "$best_checkpoint/adapter_config.json" ]]; then
    echo "ERROR: invalid best checkpoint for seed $seed: $best_checkpoint" >&2
    exit 1
  fi

  result_root="$RESULT_BASE/seed${seed}-$(basename "$best_checkpoint")"
  echo "[$(date '+%F %T')] seed=$seed checkpoint=$best_checkpoint result=$result_root"

  CODE_DIR="$CODE_DIR" \
  CKPT_DIR="$best_checkpoint" \
  RESULT_ROOT="$result_root" \
  bash "$CODE_DIR/full_beir.sh"
done

echo "[$(date '+%F %T')] All three BEIR-8 evaluations completed"
