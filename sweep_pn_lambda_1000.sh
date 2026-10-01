#!/usr/bin/env bash
# Train PN-lambda candidates from the same frozen DeAR+KK+SLL initialization.
set -euo pipefail
cd /root/dear_kk_mse_new

STUDENT_INIT="${STUDENT_INIT:-/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new}"
PN_SAFE_REFERENCE="${PN_SAFE_REFERENCE:-$STUDENT_INIT}"
PN_LAMBDA_WEIGHTS="${PN_LAMBDA_WEIGHTS:-0.02 0.05 0.10}"

for weight in $PN_LAMBDA_WEIGHTS; do
  tag="${weight/./p}"
  STUDENT_INIT="$STUDENT_INIT" \
  PN_SAFE_REFERENCE="$PN_SAFE_REFERENCE" \
  LAMBDA_PN_SAFE=0.1 \
  LAMBDA_PN_LAMBDA="$weight" \
  MAX_STEPS=1000 \
  TRAIN_OUTPUT="/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new-pngap-pnsafe-pnlambda-l${tag}-1000" \
    bash train_pn_lambda_1000.sh
done
