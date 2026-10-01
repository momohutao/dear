#!/usr/bin/env bash
# Continue from SLL+KK DeAR; do not restart from the Llama base model.
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CODE_DIR=/root/dear_kk_mse_new
BASE_CKPT=/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-setwise-sll-new
RUN_NAME=dear-p-3b-sll-kk-setwise-sll-continue500
OUTPUT_DIR=/hy-tmp/DeAR/outputs/$RUN_NAME
LOG_FILE=/hy-tmp/DeAR/logs/$RUN_NAME.log
MAX_STEPS="${MAX_STEPS:-500}"

mkdir -p /hy-tmp/DeAR/outputs /hy-tmp/DeAR/logs
cd "$CODE_DIR"

python -m py_compile pointwise_reranker/setwise_head.py pointwise_reranker/modeling.py pointwise_reranker/trainer.py
test -s "$BASE_CKPT/adapter_config.json"
test -s "$BASE_CKPT/adapter_model.bin"

deepspeed --include localhost:0 pointwise_reranker/reranker_train.py \
  --deepspeed ./config/ds_config_setwise.json \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$BASE_CKPT" \
  --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --loss_type softmax_ce --temperature 2.0 --alpha 0.1 \
  --use_relation_distill true --lambda_rat 0.02 --lambda_rvr 0 \
  --relation_temperature 2 --relation_heads 8 \
  --student_relation_layer 27 --teacher_relation_layers 39 \
  --relation_use_rope true --relation_eps 1e-8 \
  --lambda_margin_sll 0 \
  --use_setwise_head true --lambda_sll 0.1 \
  --set_size 5 --set_heads 4 --set_layers 2 --set_hidden_dim 256 \
  --set_teacher_temperature 2.0 --set_student_temperature 2.0 \
  --setwise_head_learning_rate 1e-4 \
  --dataset_name json --train_dir /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --fp16 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 \
  --train_n_passages 5 --learning_rate 1e-6 \
  --q_max_len 32 --p_max_len 196 --max_steps "$MAX_STEPS" \
  --logging_steps 50 --save_strategy no --seed 42 --data_seed 42 \
  --dataset_proc_num 1 --dataloader_num_workers 2 --full_determinism true \
  --report_to none --overwrite_output_dir 2>&1 | tee "$LOG_FILE"

test -s "$OUTPUT_DIR/adapter_model.bin"
test -s "$OUTPUT_DIR/setwise_head.bin"
test -s "$OUTPUT_DIR/setwise_head_config.json"
