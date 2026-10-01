#!/usr/bin/env bash
# Evaluate trained PN-lambda weight candidates directly on DL19.
set -euo pipefail

cd /root/dear_kk_mse_new
PN_LAMBDA_WEIGHTS="${PN_LAMBDA_WEIGHTS:-0.02 0.05 0.10}"

for weight in $PN_LAMBDA_WEIGHTS; do
  tag="${weight/./p}"
  ckpt="/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new-pngap-pnsafe-pnlambda-l${tag}-1000"
  result_dir="results/$(basename "$ckpt")-raw-dl19"
  echo "[DL19] lambda_pn_lambda=$weight checkpoint=$ckpt"
  CKPT="$ckpt" RESULT_DIR="$result_dir" bash eval_rg_dal_dl19.sh
done

echo
echo "DL19 summaries:"
for weight in $PN_LAMBDA_WEIGHTS; do
  tag="${weight/./p}"
  ckpt="/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new-pngap-pnsafe-pnlambda-l${tag}-1000"
  result_file="results/$(basename "$ckpt")-raw-dl19/evaluation_results.log"
  echo "lambda_pn_lambda=$weight"
  cat "$result_file"
done
