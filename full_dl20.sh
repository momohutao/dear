#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export DEAR_ROOT=/hy-tmp/DeAR
export CODE_DIR=/root/dear_kk_mse_new
export RUN_NAME=dear-p-3b-sll-kk-new
# DeAR原始方法1000-step checkpoint
# export CKPT_DIR=/hy-tmp/DeAR/outputs/dear-p-3b-original-1k-fixed
export CKPT_DIR=/hy-tmp/DeAR/outputs/${RUN_NAME}
# DL20输入与标签
export INPUT_FILE=/hy-tmp/DeAR/data/bm25_beir_dl19_20/dl20.jsonl
export QRELS_FILE=/hy-tmp/DeAR/data/qrels/2020qrels-pass.txt

# 使用全新结果目录，避免复用DL19或其他checkpoint结果
# export RESULT_DIR=/root/dear74.45/results/dear_kk_dl20
export RESULT_DIR=/root/dear_kk_mse_new/results/${RUN_NAME}-tta
export RAW_FILE="$RESULT_DIR/dl20.txt"
export TREC_FILE="$RESULT_DIR/ranked_dl20.trec"
export EVAL_FILE="$RESULT_DIR/evaluation_results.log"
export RUN_LOG="$RESULT_DIR/inference.log"

export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

mkdir -p \
  "$RESULT_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TMPDIR"

cd "$CODE_DIR"

test -s "$CKPT_DIR/adapter_config.json"

if [[ ! -s "$CKPT_DIR/adapter_model.bin" ]] &&
   [[ ! -s "$CKPT_DIR/adapter_model.safetensors" ]]; then
  echo "ERROR: adapter weights are missing in $CKPT_DIR"
  exit 1
fi

test -s "$INPUT_FILE"
test -s "$QRELS_FILE"

test -s /hy-tmp/DeAR/models/Llama-3.2-3B/config.json
test -s /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher/config.json

python -m py_compile \
  pointwise_reranker/inference.py \
  pointwise_reranker/modeling.py \
  pointwise_reranker/data.py \
  pointwise_reranker/convert_result_to_trec.py

echo "===== DeAR query-level TTA: DL20 inference starts ====="

# 若推理结果已存在，则跳过昂贵的模型推理
if [[ ! -s "$RAW_FILE" ]]; then
  CUDA_VISIBLE_DEVICES=0 python \
    pointwise_reranker/inference.py \
    --output_dir "$RESULT_DIR/temp_dl20" \
    --model_name_or_path "$CKPT_DIR" \
    --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
    --teacher_model_name_or_path \
      /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
    --encode_in_path "$INPUT_FILE" \
    --encoded_save_path "$RAW_FILE" \
    --dataset_name json \
    --loss_type softmax_ce \
    --tta_enabled true \
    --tta_rank 4 \
    --tta_gamma 0.05 \
    --tta_steps 3 \
    --tta_learning_rate 1e-3 \
    --tta_top_k 100 \
    --tta_n_pos 5 \
    --tta_n_neg 20 \
    --tta_temperature 0.5 \
    --tta_anchor_weight 1.0 \
    --fp16 \
    --per_device_eval_batch_size 32 \
    --q_max_len 32 \
    --p_max_len 196 \
    --dataset_proc_num 2 \
    --dataloader_num_workers 2 \
    --overwrite_output_dir \
    2>&1 | tee "$RUN_LOG"
else
  echo "DL20 raw result exists; skipping inference"
fi

test -s "$RAW_FILE"

echo "===== Convert to TREC format ====="

python \
  pointwise_reranker/convert_result_to_trec.py \
  --input "$RAW_FILE" \
  --output "$TREC_FILE"

test -s "$TREC_FILE"

echo "===== Evaluate DL20 nDCG@10 ====="

python -m pyserini.eval.trec_eval \
  -c \
  -m ndcg_cut.10 \
  "$QRELS_FILE" \
  "$TREC_FILE" \
  2>&1 | tee "$EVAL_FILE"

echo "===== Finished ====="

grep "ndcg_cut_10" "$EVAL_FILE"
