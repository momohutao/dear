#!/usr/bin/env bash
# Train CE + KD + RAT/KK + teacher-gated PN-gap + PN-lambda from raw Llama-3.2-3B.
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /root/dear_kk_mse_new
DAL_INIT_CKPT="${DAL_INIT_CKPT:-/hy-tmp/DeAR/models/Llama-3.2-3B}"
DAL_OUTPUT="${DAL_OUTPUT:-/hy-tmp/DeAR/outputs/dear-p-3b-ce-kd-kk-pngap-pnlambda-no-sll-1000}"
MAX_STEPS="${MAX_STEPS:-1000}"
mkdir -p /hy-tmp/DeAR/logs

if test -s "$DAL_INIT_CKPT/adapter_config.json"; then
  LEARNING_RATE="${LEARNING_RATE:-1e-6}"
  echo "[train] Resuming DeAR adapter: $DAL_INIT_CKPT"
else
  LEARNING_RATE="${LEARNING_RATE:-5e-5}"
  test -f "$DAL_INIT_CKPT/config.json"
  echo "[train] Starting new LoRA + classification head from Llama base: $DAL_INIT_CKPT"
fi
python -m py_compile pointwise_reranker/arguments.py pointwise_reranker/modeling.py pointwise_reranker/trainer.py
# alpha controls global KD; relation_distill/RAT is the KK term.
# PN-gap is enabled through DAL and PN-lambda uses swap-weighted labels.
deepspeed --include localhost:0 pointwise_reranker/reranker_train.py \
  --deepspeed ./config/ds_config_setwise.json --output_dir "$DAL_OUTPUT" \
  --model_name_or_path "$DAL_INIT_CKPT" --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --loss_type softmax_ce --temperature 2.0 --alpha 0.1 \
  --use_relation_distill true --lambda_rat 0.02 --lambda_rvr 0 --relation_temperature 2.0 --relation_heads 8 \
  --student_relation_layer 27 --teacher_relation_layers 39 --relation_use_rope true --relation_eps 1e-8 \
  --use_setwise_head false \
  --use_dal true --use_pn_gap true --lambda_dal 0.01 \
  --dal_pn_weight 1.0 --dal_nn_weight 0 \
  --dal_pn_teacher_weight 0.10 --dal_pn_teacher_temperature 0.25 \
  --dal_gap_temperature 0.25 --dal_teacher_confidence_scale 0.5 --dal_teacher_margin_cap 2.0 \
  --dal_warmup_ratio 0 \
  --use_pn_lambda true --lambda_pn_lambda "${LAMBDA_PN_LAMBDA:-0.1}" \
  --pn_lambda_temperature "${PN_LAMBDA_TAU:-0.25}" \
  --dataset_name json --train_dir /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --fp16 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --train_n_passages 8 --learning_rate "$LEARNING_RATE" \
  --q_max_len 32 --p_max_len 196 --max_steps "$MAX_STEPS" --logging_steps 25 --save_strategy no \
  --seed 42 --data_seed 42 --dataset_proc_num 1 --dataloader_num_workers 2 --report_to none --overwrite_output_dir \
  2>&1 | tee "/hy-tmp/DeAR/logs/$(basename "$DAL_OUTPUT").log"

test -s "$DAL_OUTPUT/adapter_config.json"
test -f "$DAL_OUTPUT/adapter_model.bin" -o -f "$DAL_OUTPUT/adapter_model.safetensors"
