#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p
cd /root/dear-sll-vv

export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p /hy-tmp/DeAR/outputs /hy-tmp/DeAR/logs "$TMPDIR"

# Each seed uses all four GPUs; seeds run sequentially.
if (( $# == 0 )); then set -- 42 43 44; fi
for seed in "$@"; do
  case "$seed" in 42|43|44) ;; *) echo "Expected seed 42, 43 or 44" >&2; exit 1 ;; esac
  run_name="dear-qk-mvp-beir8-seed${seed}-max6000-$(date +%Y%m%d-%H%M%S)"
  output_dir="/hy-tmp/DeAR/outputs/$run_name"
  if [[ -e "$output_dir" ]]; then
    echo "Output already exists: $output_dir" >&2
    exit 1
  fi
  export PYTHONHASHSEED="$seed"
  deepspeed --include localhost:0,1,2,3 \
    pointwise_reranker/reranker_train.py \
    --deepspeed ./config/ds_config.json \
    --output_dir "$output_dir" \
    --model_name_or_path /hy-tmp/DeAR/models/Llama-3.2-3B \
    --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
    --temperature 2 --alpha 0.1 --loss_type softmax_ce \
    --lambda_margin_mse 0.01 --use_relation_distill true \
    --lambda_rat 0.02 --lambda_rvr 0 --lambda_down 0.01 \
    --relation_temperature 2 --relation_heads 8 \
    --student_relation_layer 27 --teacher_relation_layers 39 \
    --relation_use_rope true --relation_eps 1e-8 \
    --dataset_name json --train_dir /hy-tmp/DeAR/data/MVP-train \
    --fp16 --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
    --train_n_passages 5 --learning_rate 1e-4 --q_max_len 32 --p_max_len 196 \
    --max_steps 6000 --logging_steps 20 --save_strategy no \
    --early_stop_metric beir8 --early_stop_eval_steps 500 \
    --early_stop_patience "${EARLY_STOP_PATIENCE:-3}" --early_stop_min_delta 0 \
    --early_stop_eval_batch_size 32 \
    --seed "$seed" --data_seed "$seed" \
    --dataset_proc_num 8 --dataloader_num_workers 4 \
    --dataloader_pin_memory true --dataloader_persistent_workers true \
    --ddp_timeout 7200 --full_determinism false --report_to none \
    2>&1 | tee "/hy-tmp/DeAR/logs/$run_name.log"
done
