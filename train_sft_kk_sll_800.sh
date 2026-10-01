#!/usr/bin/env bash
# Stage 1A: ordinary DeAR + KK + contextual SLL training.  This does load the
# RankLLaMA teacher; Stage 1B below does not.
set -euo pipefail
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p
export HF_HOME=/hy-tmp/DeAR/cache/hf HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd /root/dear_kk_mse_new

# Override these paths rather than editing the script, e.g.
# SFT_INIT_CKPT=... SFT_OUTPUT=... MAX_STEPS=1000 ./train_sft_kk_sll_800.sh
SFT_INIT_CKPT="${SFT_INIT_CKPT:-/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new}"
SFT_OUTPUT="${SFT_OUTPUT:-/hy-tmp/DeAR/outputs/dear-p-3b-sft-kk-sll-800}"
MAX_STEPS="${MAX_STEPS:-800}"
mkdir -p /hy-tmp/DeAR/logs

deepspeed --include localhost:0 pointwise_reranker/reranker_train.py \
  --deepspeed ./config/ds_config_setwise.json --output_dir "$SFT_OUTPUT" \
  --model_name_or_path "$SFT_INIT_CKPT" --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --loss_type softmax_ce --temperature 2.0 --alpha 0.1 \
  --use_relation_distill true --lambda_rat 0.02 --lambda_rvr 0 --relation_temperature 2 --relation_heads 8 \
  --student_relation_layer 27 --teacher_relation_layers 39 --relation_use_rope true --relation_eps 1e-8 \
  --lambda_margin_sll 0 --use_setwise_head true --lambda_sll 0.1 --set_size 5 --set_heads 4 --set_layers 2 --set_hidden_dim 256 \
  --set_teacher_temperature 2.0 --set_student_temperature 2.0 --setwise_head_learning_rate 1e-4 \
  --dataset_name json --train_dir /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --fp16 --per_device_train_batch_size 2 --gradient_accumulation_steps 1 --train_n_passages 5 --learning_rate 1e-6 \
  --q_max_len 32 --p_max_len 196 --max_steps "$MAX_STEPS" --logging_steps 50 --save_strategy no \
  --seed 42 --data_seed 42 --dataset_proc_num 1 --dataloader_num_workers 2 --report_to none --overwrite_output_dir \
  2>&1 | tee "/hy-tmp/DeAR/logs/$(basename "$SFT_OUTPUT").log"
