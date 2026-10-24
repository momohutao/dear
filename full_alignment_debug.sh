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

export CODE_DIR=/root/dear_mse

export RUN_NAME=dear-alignment-debug


export CKPT_DIR=/hy-tmp/DeAR/outputs/${RUN_NAME}

export LOG_FILE=/hy-tmp/DeAR/logs/${RUN_NAME}.log

export RESULT_DIR=/root/dear_mse/results/${RUN_NAME}

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

echo "===== Stage 1 training starts ====="

deepspeed --include localhost:0,1,2,3 \
  pointwise_reranker/reranker_train.py \
  --deepspeed ./config/ds_config.json \
  --output_dir "$CKPT_DIR" \
  --model_name_or_path /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --temperature 2 \
  --alpha 0.1 \
  --loss_type softmax_ce \
  --lambda_margin_mse 0.01 \
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
  --train_dir /hy-tmp/DeAR/data/msmarco-passage/train.align-smoke.jsonl.gz \
  --fp16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --train_n_passages 2 \
  --learning_rate 1e-4 \
  --q_max_len 32 \
  --p_max_len 196 \
  --max_steps 1000 \
  --logging_steps 20 \
  --save_strategy no \
  --adapter_save_steps 1000 \
  --seed 42 \
  --data_seed 42 \
  --dataset_proc_num 1 \
  --dataloader_num_workers 0 \
  --full_determinism true \
  --report_to none \
  --overwrite_output_dir \
  2>&1 | tee "$LOG_FILE"

# 训练命令必须成功结束，且保存出真实 adapter，才允许测试。
test -s "$CKPT_DIR/adapter_config.json"
test -s "$CKPT_DIR/adapter_model.bin"

echo "===== Adapter saved; DL19 inference starts ====="

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