#!/usr/bin/env bash
# Inference-only continuous RankLLaMA distribution TTA; never trains or writes
# into the source DeAR checkpoint.
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CODE_DIR=/root/dear_kk_mse_new
CKPT=/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new
MODE="${MODE:-global_local_gate}"
K="${TTA_CANDIDATE_K:-30}"
GAMMA="${TTA_GAMMA:-0.10}"
STEPS="${TTA_STEPS:-10}"
LR="${TTA_LR:-1e-2}"
LAMBDA_LOCAL="${TTA_LAMBDA_LOCAL:-1.0}"
USE_GATE="${TTA_USE_CONFIDENCE_GATE:-true}"
MIN_CONF="${TTA_MIN_TEACHER_CONFIDENCE:-0.0}"
CACHE_PATH="${TTA_TEACHER_CACHE_PATH:-/hy-tmp/DeAR/cache/teacher_scores/rankllama_dl19_top30.jsonl}"

case "$MODE" in
  global) LAMBDA_LOCAL=0; USE_GATE=false ;;
  global_local) USE_GATE=false ;;
  global_local_gate) : ;;
  *) echo "MODE must be global, global_local, or global_local_gate" >&2; exit 2 ;;
esac

RESULT_DIR=$CODE_DIR/results/dear-p-3b-sll-kk-new-distribution-tta-${MODE}-k${K}-g${GAMMA}-s${STEPS}
mkdir -p "$RESULT_DIR"
cd "$CODE_DIR"

python -m py_compile pointwise_reranker/inference.py pointwise_reranker/modeling.py pointwise_reranker/distribution_tta.py pointwise_reranker/teacher_score_cache.py

python pointwise_reranker/inference.py \
  --output_dir "$RESULT_DIR/temp" \
  --model_name_or_path "$CKPT" \
  --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --encode_in_path /hy-tmp/DeAR/data/bm25_beir_dl19_20/dl19.jsonl \
  --encoded_save_path "$RESULT_DIR/dl19.txt" --dataset_name json --loss_type softmax_ce \
  --use_distribution_tta true --tta_candidate_k "$K" --tta_gamma "$GAMMA" \
  --tta_steps "$STEPS" --tta_lr "$LR" \
  --tta_teacher_temperature 2.0 --tta_student_temperature 2.0 \
  --tta_lambda_local "$LAMBDA_LOCAL" --tta_window_size 10 --tta_window_stride 5 \
  --tta_anchor_weight 0.1 --tta_u_norm_weight 1e-4 \
  --tta_use_confidence_gate "$USE_GATE" --tta_min_teacher_confidence "$MIN_CONF" \
  --tta_early_stop_patience 3 --tta_early_stop_tolerance 1e-5 \
  --tta_teacher_cache_path "$CACHE_PATH" --tta_log_diagnostics true \
  --fp16 --per_device_eval_batch_size 32 --q_max_len 32 --p_max_len 196 \
  --dataset_proc_num 1 --dataloader_num_workers 0 --overwrite_output_dir \
  2>&1 | tee "$RESULT_DIR/inference.log"

python pointwise_reranker/convert_result_to_trec.py --input "$RESULT_DIR/dl19.txt" --output "$RESULT_DIR/ranked_dl19.trec"
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 /hy-tmp/DeAR/data/qrels/2019qrels-pass.txt \
  "$RESULT_DIR/ranked_dl19.trec" | tee "$RESULT_DIR/evaluation_results.log"
