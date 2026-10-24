#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export DEAR_ROOT=/hy-tmp/DeAR
export CODE_DIR=/root/dear-sll-vv
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export RUN_NAME="${RUN_NAME:-dear-p-3b-SLL-kk-downkl-seed123-1k}"
export EVAL_SET="${EVAL_SET:-dl20}"
EVAL_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# DeAR原始方法1000-step checkpoint
# export CKPT_DIR=/hy-tmp/DeAR/outputs/dear-p-3b-original-1k-fixed
export CKPT_DIR="${CKPT_DIR:-/hy-tmp/DeAR/outputs/${RUN_NAME}}"
case "$EVAL_SET" in
  dl19)
    export INPUT_FILE=/hy-tmp/DeAR/data/bm25_beir_dl19_20/dl19.jsonl
    export QRELS_FILE=/hy-tmp/DeAR/data/qrels/2019qrels-pass.txt
    ;;
  dl20)
    export INPUT_FILE=/hy-tmp/DeAR/data/bm25_beir_dl19_20/dl20.jsonl
    export QRELS_FILE=/hy-tmp/DeAR/data/qrels/qrels.dl20-passage.txt
    ;;
  *)
    echo "ERROR: EVAL_SET must be dl19 or dl20, got: $EVAL_SET"
    exit 1
    ;;
esac
# 使用全新结果目录，避免复用DL19或其他checkpoint结果
# export RESULT_DIR=/root/dear74.45/results/dear_kk_dl20
export RESULT_DIR=/root/dear_mse/results/${RUN_NAME}
export RAW_FILE="$RESULT_DIR/${EVAL_SET}.txt"
export TREC_FILE="$RESULT_DIR/ranked_${EVAL_SET}.trec"
export EVAL_FILE="$RESULT_DIR/${EVAL_SET}_evaluation_results.log"
export RUN_LOG="$RESULT_DIR/${EVAL_SET}_inference.log"

export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:${LD_LIBRARY_PATH:-}"

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

echo "===== KK ${RUN_NAME}: ${EVAL_SET} inference starts ====="

# 若推理结果已存在，则跳过昂贵的模型推理
if [[ ! -s "$RAW_FILE" ]]; then
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICES%%,*}" python \
    pointwise_reranker/inference.py \
    --output_dir "$RESULT_DIR/temp_${EVAL_SET}" \
    --model_name_or_path "$CKPT_DIR" \
    --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
    --teacher_model_name_or_path \
      /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
    --encode_in_path "$INPUT_FILE" \
    --encoded_save_path "$RAW_FILE" \
    --dataset_name json \
    --loss_type softmax_ce \
    --fp16 \
    --per_device_eval_batch_size 32 \
    --q_max_len 32 \
    --p_max_len 196 \
    --dataset_proc_num 2 \
    --dataloader_num_workers 2 \
    --overwrite_output_dir \
    2>&1 | tee "$RUN_LOG"
else
  echo "${EVAL_SET} raw result exists; skipping inference"
fi

test -s "$RAW_FILE"

echo "===== Convert to TREC format ====="

python \
  pointwise_reranker/convert_result_to_trec.py \
  --input "$RAW_FILE" \
  --output "$TREC_FILE"

test -s "$TREC_FILE"

echo "===== Evaluate ${EVAL_SET} nDCG@10 ====="

python -m pyserini.eval.trec_eval \
  -c \
  -m ndcg_cut.10 \
  "$QRELS_FILE" \
  "$TREC_FILE" \
  2>&1 | tee "$EVAL_FILE"

echo "===== Finished ====="

grep "ndcg_cut_10" "$EVAL_FILE"
