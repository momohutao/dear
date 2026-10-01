#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export DEAR_ROOT=/hy-tmp/DeAR
export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
# 固定Python与CUDA计算
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

export CODE_DIR=/root/dear_kk_mse_new
export RUN_NAME=dear-p-3b-sll-kk-new
export CKPT_DIR=/hy-tmp/DeAR/outputs/${RUN_NAME}
export LOG_FILE=/hy-tmp/DeAR/logs/${RUN_NAME}.log
export RESULT_DIR=/root/dear_kk_mse_new/results/${RUN_NAME}

# Optional inference-only RankLLaMA distribution TTA.  It is disabled by
# default, so this training-and-baseline script keeps its original behavior.
export USE_DISTRIBUTION_TTA="${USE_DISTRIBUTION_TTA:-false}"
export USE_ENTROPY_HARD_GATE_TTA="${USE_ENTROPY_HARD_GATE_TTA:-false}"
export TTA_CANDIDATE_K="${TTA_CANDIDATE_K:-30}"
export TTA_GAMMA="${TTA_GAMMA:-0.10}"
export TTA_STEPS="${TTA_STEPS:-10}"
export TTA_LR="${TTA_LR:-1e-2}"
export TTA_TEACHER_TEMPERATURE="${TTA_TEACHER_TEMPERATURE:-2.0}"
export TTA_STUDENT_TEMPERATURE="${TTA_STUDENT_TEMPERATURE:-2.0}"
export TTA_LAMBDA_LOCAL="${TTA_LAMBDA_LOCAL:-1.0}"
export TTA_WINDOW_SIZE="${TTA_WINDOW_SIZE:-10}"
export TTA_WINDOW_STRIDE="${TTA_WINDOW_STRIDE:-5}"
export TTA_ANCHOR_WEIGHT="${TTA_ANCHOR_WEIGHT:-0.1}"
export TTA_U_NORM_WEIGHT="${TTA_U_NORM_WEIGHT:-1e-4}"
export TTA_USE_CONFIDENCE_GATE="${TTA_USE_CONFIDENCE_GATE:-true}"
export TTA_MIN_TEACHER_CONFIDENCE="${TTA_MIN_TEACHER_CONFIDENCE:-0.0}"
export TTA_EARLY_STOP_PATIENCE="${TTA_EARLY_STOP_PATIENCE:-3}"
export TTA_EARLY_STOP_TOLERANCE="${TTA_EARLY_STOP_TOLERANCE:-1e-5}"
export TTA_TEACHER_CACHE_PATH="${TTA_TEACHER_CACHE_PATH:-/hy-tmp/DeAR/cache/teacher_scores/rankllama_dl19_top30.jsonl}"
export TTA_ENTROPY_TEMPERATURE="${TTA_ENTROPY_TEMPERATURE:-2.0}"
export TTA_ENTROPY_THRESHOLD="${TTA_ENTROPY_THRESHOLD:-0.0}"
export TTA_FORCE_ADAPTED="${TTA_FORCE_ADAPTED:-false}"
export TTA_SINGLE_FOLD_GATE="${TTA_SINGLE_FOLD_GATE:-false}"

mkdir -p /hy-tmp/DeAR/outputs
mkdir -p /hy-tmp/DeAR/logs
mkdir -p /hy-tmp/DeAR/tmp
mkdir -p "$RESULT_DIR"

cd "$CODE_DIR"

python -m py_compile \
  pointwise_reranker/reranker_train.py \
  pointwise_reranker/trainer.py \
  pointwise_reranker/modeling.py \
  pointwise_reranker/region_relation.py

echo "===== KK + Margin Smooth-L1 training starts ====="

deepspeed --include localhost:0 \
  pointwise_reranker/reranker_train.py \
  --deepspeed ./config/ds_config.json \
  --output_dir "$CKPT_DIR" \
  --model_name_or_path /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --temperature 2 \
  --alpha 0.1 \
  --loss_type softmax_ce \
  --lambda_margin_sll 0.01 \
  --margin_sll_beta 1.0 \
  --use_relation_distill true \
  --lambda_rat 0.02 \
  --lambda_rvr 0 \
  --relation_temperature 2 \
  --relation_heads 8 \
  --student_relation_layer 27 \
  --teacher_relation_layers 39 \
  --relation_use_rope true \
  --relation_eps 1e-8 \
  --dataset_name json \
  --train_dir /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --fp16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --train_n_passages 2 \
  --learning_rate 1e-4 \
  --q_max_len 32 \
  --p_max_len 196 \
  --max_steps 1000 \
  --logging_steps 100 \
  --save_strategy no \
  --seed 42 \
  --data_seed 42 \
  --dataset_proc_num 1 \
  --dataloader_num_workers 2 \
  --full_determinism true \
  --report_to none \
  --overwrite_output_dir \
  2>&1 | tee "$LOG_FILE"

test -s "$CKPT_DIR/adapter_config.json"
test -s "$CKPT_DIR/adapter_model.bin"

echo "===== KK + Margin Smooth-L1 training complete; DL19 inference starts ====="

DISTRIBUTION_TTA_ARGS=()
if [[ "$USE_DISTRIBUTION_TTA" == "true" ]]; then
  DISTRIBUTION_TTA_ARGS=(
    --use_distribution_tta true
    --tta_candidate_k "$TTA_CANDIDATE_K" --tta_gamma "$TTA_GAMMA"
    --tta_steps "$TTA_STEPS" --tta_lr "$TTA_LR"
    --tta_teacher_temperature "$TTA_TEACHER_TEMPERATURE"
    --tta_student_temperature "$TTA_STUDENT_TEMPERATURE"
    --tta_lambda_local "$TTA_LAMBDA_LOCAL"
    --tta_window_size "$TTA_WINDOW_SIZE" --tta_window_stride "$TTA_WINDOW_STRIDE"
    --tta_anchor_weight "$TTA_ANCHOR_WEIGHT" --tta_u_norm_weight "$TTA_U_NORM_WEIGHT"
    --tta_use_confidence_gate "$TTA_USE_CONFIDENCE_GATE"
    --tta_min_teacher_confidence "$TTA_MIN_TEACHER_CONFIDENCE"
    --tta_early_stop_patience "$TTA_EARLY_STOP_PATIENCE"
    --tta_early_stop_tolerance "$TTA_EARLY_STOP_TOLERANCE"
    --tta_teacher_cache_path "$TTA_TEACHER_CACHE_PATH"
  )
fi

ENTROPY_HARD_GATE_TTA_ARGS=()
if [[ "$USE_ENTROPY_HARD_GATE_TTA" == "true" ]]; then
  ENTROPY_HARD_GATE_TTA_ARGS=(
    --use_entropy_hard_gate_tta true
    --tta_candidate_k "$TTA_CANDIDATE_K" --tta_gamma "$TTA_GAMMA"
    --tta_steps "$TTA_STEPS" --tta_lr "$TTA_LR"
    --tta_teacher_temperature "$TTA_TEACHER_TEMPERATURE"
    --tta_student_temperature "$TTA_STUDENT_TEMPERATURE"
    --tta_entropy_temperature "$TTA_ENTROPY_TEMPERATURE"
    --tta_entropy_threshold "$TTA_ENTROPY_THRESHOLD"
    --tta_anchor_weight "$TTA_ANCHOR_WEIGHT" --tta_u_norm_weight "$TTA_U_NORM_WEIGHT"
    --tta_teacher_cache_path "$TTA_TEACHER_CACHE_PATH"
    --tta_force_adapted "$TTA_FORCE_ADAPTED"
    --tta_single_fold_gate "$TTA_SINGLE_FOLD_GATE"
    --tta_log_diagnostics true
  )
fi

CUDA_VISIBLE_DEVICES=0 python pointwise_reranker/inference.py \
  --output_dir "$RESULT_DIR/temp_dl19" \
  --model_name_or_path "$CKPT_DIR" \
  --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --encode_in_path /hy-tmp/DeAR/data/bm25_beir_dl19_20/dl19.jsonl \
  --encoded_save_path "$RESULT_DIR/dl19.txt" \
  --dataset_name json \
  --loss_type softmax_ce \
  --fp16 \
  --per_device_eval_batch_size 32 \
  --q_max_len 32 \
  --p_max_len 196 \
  --dataloader_num_workers 0 \
  "${DISTRIBUTION_TTA_ARGS[@]}" \
  "${ENTROPY_HARD_GATE_TTA_ARGS[@]}" \
  --overwrite_output_dir

test -s "$RESULT_DIR/dl19.txt"

echo "===== Convert to TREC format ====="

python pointwise_reranker/convert_result_to_trec.py \
  --input "$RESULT_DIR/dl19.txt" \
  --output "$RESULT_DIR/ranked_dl19.trec"

echo "===== Evaluate DL19 nDCG@10 ====="

python -m pyserini.eval.trec_eval \
  -c \
  -m ndcg_cut.10 \
  /hy-tmp/DeAR/data/qrels/2019qrels-pass.txt \
  "$RESULT_DIR/ranked_dl19.trec" \
  | tee "$RESULT_DIR/evaluation_results.log"

echo "===== Finished ====="
grep "ndcg_cut_10" "$RESULT_DIR/evaluation_results.log"
