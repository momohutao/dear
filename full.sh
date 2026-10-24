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
# 默认使用全部四张卡；外部已设置CUDA_VISIBLE_DEVICES时尊重外部设置。
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# 默认优先吞吐；FULL_DETERMINISM=true 可恢复严格确定性。
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED:-42}}"
export FULL_DETERMINISM="${FULL_DETERMINISM:-false}"
if [[ "$FULL_DETERMINISM" == "true" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PERSISTENT_WORKERS=false
if (( ${DATALOADER_WORKERS:-4} > 0 )); then
  PERSISTENT_WORKERS=true
fi
# pyserini/pyjnius needs an explicit JVM path on this host.
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"

export CODE_DIR=/root/dear-sll-vv

export RUN_NAME="${RUN_NAME:-dear-p-3b-SLL-kk-downkl-0.01-1k-mvp}"
export TRAIN_DIR="${TRAIN_DIR:-/hy-tmp/DeAR/data/MVP-train}"

export CKPT_DIR=/hy-tmp/DeAR/outputs/${RUN_NAME}

export LOG_FILE=/hy-tmp/DeAR/logs/${RUN_NAME}.log

export RESULT_DIR=/root/dear_mse/results/${RUN_NAME}

mkdir -p /hy-tmp/DeAR/outputs
mkdir -p /hy-tmp/DeAR/logs
mkdir -p /hy-tmp/DeAR/tmp
mkdir -p "$CKPT_DIR"
mkdir -p "$RESULT_DIR"

cd "$CODE_DIR"

python -m py_compile \
  pointwise_reranker/reranker_train.py \
  pointwise_reranker/trainer.py \
  pointwise_reranker/modeling.py \
  pointwise_reranker/region_relation.py

echo "===== Training starts: $RUN_NAME ====="

# 从基础模型重新训练；DeepSpeed显式启动四个进程。
# 5个候选在32GB V100上默认每卡2个query、累积2步，维持有效query batch=16。
# 首次tokenization只由主进程执行，8个CPU worker可缩短四卡启动等待。
deepspeed --num_gpus="${NUM_GPUS:-4}" pointwise_reranker/reranker_train.py \
  --deepspeed ./config/ds_config.json \
  --output_dir "$CKPT_DIR" \
  --model_name_or_path /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --temperature 2 \
  --alpha 0.1 \
  --loss_type softmax_ce \
  --lambda_margin_mse 0.01 \
  --use_relation_distill true \
  --lambda_rat "${LAMBDA_RAT:-0}" \
  --lambda_rvr "${LAMBDA_KK:-0.02}" \
  --lambda_down "${LAMBDA_DOWN:-0.01}" \
  --relation_temperature 2 \
  --relation_heads 8 \
  --student_relation_layer 27 \
  --teacher_relation_layers 39 \
  --relation_use_rope true \
  --relation_eps 1e-8 \
  --dataset_name json \
  --train_dir "$TRAIN_DIR" \
  --fp16 \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS:-2}" \
  --train_n_passages "${TRAIN_N_PASSAGES:-5}" \
  --learning_rate 1e-4 \
  --q_max_len 32 \
  --p_max_len 196 \
  --max_steps "${MAX_STEPS:-1000}" \
  --logging_steps 20 \
  --save_strategy no \
  --adapter_save_steps "${ADAPTER_SAVE_STEPS:-1000}" \
  --early_stop_eval_steps "${EARLY_STOP_EVAL_STEPS:-0}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE:-3}" \
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA:-0}" \
  --early_stop_eval_batch_size "${EARLY_STOP_EVAL_BATCH_SIZE:-32}" \
  --seed "${SEED:-42}" \
  --data_seed "${DATA_SEED:-${SEED:-42}}" \
  --dataset_proc_num "${DATASET_PROC_NUM:-8}" \
  --dataloader_num_workers "${DATALOADER_WORKERS:-4}" \
  --dataloader_pin_memory true \
  --dataloader_persistent_workers "$PERSISTENT_WORKERS" \
  --ddp_timeout "${DDP_TIMEOUT:-7200}" \
  --full_determinism "$FULL_DETERMINISM" \
  --report_to none \
  --overwrite_output_dir \
  2>&1 | tee "$LOG_FILE"

# Set TRAIN_ONLY=1 to skip both DL19 and DL20 evaluation.
if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

# 早停训练使用验证均值最高的adapter；普通训练仍使用最终adapter。
EVAL_CKPT_DIR="$CKPT_DIR"
if [[ "${EARLY_STOP_EVAL_STEPS:-0}" -gt 0 ]]; then
  if [[ ! -s "$CKPT_DIR/best-adapter/adapter_config.json" ]]; then
    echo "错误：早停最佳模型不存在：$CKPT_DIR/best-adapter" >&2
    exit 1
  fi
  EVAL_CKPT_DIR="$CKPT_DIR/best-adapter"
fi

# 训练成功且 adapter 保存完成后，才开始评测。
for adapter_file in adapter_config.json; do
  if [[ ! -s "$EVAL_CKPT_DIR/$adapter_file" ]]; then
    echo "错误：模型文件不存在或为空：$EVAL_CKPT_DIR/$adapter_file" >&2
    exit 1
  fi
done
if [[ ! -s "$EVAL_CKPT_DIR/adapter_model.bin" && ! -s "$EVAL_CKPT_DIR/adapter_model.safetensors" ]]; then
  echo "错误：adapter 权重不存在：$EVAL_CKPT_DIR" >&2
  exit 1
fi

echo "===== Adapter saved; DL19 inference starts ====="

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" python pointwise_reranker/inference.py \
  --output_dir "$RESULT_DIR/temp_dl19" \
  --model_name_or_path "$EVAL_CKPT_DIR" \
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

echo "===== DL19 finished ====="
grep "ndcg_cut_10" "$RESULT_DIR/evaluation_results.log"

echo "===== DL20 evaluation starts ====="
CKPT_DIR="$EVAL_CKPT_DIR" EVAL_SET=dl20 bash "$CODE_DIR/full_dl20.sh"

echo "===== Training + DL19 + DL20 finished ====="
