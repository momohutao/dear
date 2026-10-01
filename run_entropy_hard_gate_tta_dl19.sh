#!/usr/bin/env bash
# DL19 inference-only experiments for holdout-entropy hard routing.
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /root/dear_kk_mse_new
CKPT=/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new
MODE="${MODE:-double}"
K="${TTA_CANDIDATE_K:-30}"
GAMMA="${TTA_GAMMA:-0.10}"
STEPS="${TTA_STEPS:-10}"
LR="${TTA_LR:-1e-2}"
CACHE="${TTA_TEACHER_CACHE_PATH:-/hy-tmp/DeAR/cache/teacher_scores/rankllama_dl19_entropy_hard_gate_k30.jsonl}"
NAME="dear-p-3b-sll-kk-new-entropy-hard-${MODE}-k${K}-g${GAMMA}-s${STEPS}"
RESULT_DIR="results/${NAME}"
mkdir -p "$RESULT_DIR"

python -m py_compile pointwise_reranker/inference.py pointwise_reranker/entropy_hard_gate_tta.py pointwise_reranker/teacher_score_cache.py

TTA_ARGS=()
case "$MODE" in
  raw) ;;
  force)
    TTA_ARGS=(--use_entropy_hard_gate_tta true --tta_force_adapted true)
    ;;
  single)
    TTA_ARGS=(--use_entropy_hard_gate_tta true --tta_single_fold_gate true)
    ;;
  double)
    TTA_ARGS=(--use_entropy_hard_gate_tta true)
    ;;
  *) echo "MODE must be raw, force, single, or double" >&2; exit 2 ;;
esac

python pointwise_reranker/inference.py \
  --output_dir "$RESULT_DIR/temp" \
  --model_name_or_path "$CKPT" \
  --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --encode_in_path /hy-tmp/DeAR/data/bm25_beir_dl19_20/dl19.jsonl \
  --encoded_save_path "$RESULT_DIR/dl19.txt" --dataset_name json --loss_type softmax_ce \
  --tta_candidate_k "$K" --tta_gamma "$GAMMA" --tta_steps "$STEPS" --tta_lr "$LR" \
  --tta_teacher_temperature 2.0 --tta_student_temperature 2.0 \
  --tta_entropy_temperature 2.0 --tta_entropy_threshold 0.0 \
  --tta_anchor_weight 0.1 --tta_u_norm_weight 1e-4 \
  --tta_teacher_cache_path "$CACHE" --tta_log_diagnostics true \
  "${TTA_ARGS[@]}" \
  --fp16 --per_device_eval_batch_size 32 --q_max_len 32 --p_max_len 196 \
  --dataset_proc_num 1 --dataloader_num_workers 0 --overwrite_output_dir \
  2>&1 | tee "$RESULT_DIR/inference.log"

python pointwise_reranker/convert_result_to_trec.py --input "$RESULT_DIR/dl19.txt" --output "$RESULT_DIR/ranked_dl19.trec"
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 /hy-tmp/DeAR/data/qrels/2019qrels-pass.txt \
  "$RESULT_DIR/ranked_dl19.trec" | tee "$RESULT_DIR/evaluation_results.log"
