#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export DEAR_ROOT=/hy-tmp/DeAR
export CODE_DIR=/root/dear74.45
export CKPT_DIR=/hy-tmp/DeAR/outputs/dear-p-3b-original-1k-fixed
export RESULT_DIR=/root/dear74.45/results/dl19_dear_original_1k_fixed

export STUDENT_MODEL=/hy-tmp/DeAR/models/Llama-3.2-3B
export TEACHER_MODEL=/hy-tmp/DeAR/models/llama2-13b-rankllama-teacher

# 如自动检测不到，可在运行前显式指定：
# INPUT_FILE=/绝对路径/dl19.jsonl QRELS_FILE=/绝对路径/qrels.txt bash 本脚本
INPUT_FILE="${INPUT_FILE:-}"
QRELS_FILE="${QRELS_FILE:-}"
GPU_ID="${GPU_ID:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

pick_first_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -s "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ -z "$INPUT_FILE" ]]; then
  INPUT_FILE="$(pick_first_file \
    "$DEAR_ROOT/data/bm25_beir_dl19_20/dl19.jsonl" \
    "$DEAR_ROOT/data/bm25_beir_dl19_20/msmarco-passage-trec2019.jsonl" \
    "$DEAR_ROOT/data/dl19.jsonl" \
    "$DEAR_ROOT/data/trec_dl19.jsonl" \
    || true)"
fi

if [[ -z "$QRELS_FILE" ]]; then
  QRELS_FILE="$(pick_first_file \
    "$DEAR_ROOT/data/qrels/2019qrels-pass.txt" \
    "$DEAR_ROOT/data/qrels/qrels.dl19-passage.txt" \
    "$DEAR_ROOT/data/qrels/qrels.dl19.txt" \
    "$DEAR_ROOT/data/bm25_beir_dl19_20/2019qrels-pass.txt" \
    || true)"
fi

mkdir -p "$RESULT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

[[ -d "$CODE_DIR" ]] || { echo "ERROR: CODE_DIR 不存在：$CODE_DIR"; exit 1; }
[[ -s "$CKPT_DIR/adapter_config.json" ]] || { echo "ERROR: 缺少 adapter_config.json：$CKPT_DIR"; exit 1; }
compgen -G "$CKPT_DIR/adapter_model.*" >/dev/null || { echo "ERROR: 缺少 adapter_model.bin/safetensors：$CKPT_DIR"; exit 1; }
[[ -s "$STUDENT_MODEL/config.json" ]] || { echo "ERROR: 学生基础模型不存在：$STUDENT_MODEL"; exit 1; }
[[ -s "$TEACHER_MODEL/config.json" ]] || { echo "ERROR: 教师模型不存在：$TEACHER_MODEL"; exit 1; }

if [[ -z "$INPUT_FILE" || ! -s "$INPUT_FILE" ]]; then
  echo "ERROR: 未找到 DL19 推理输入 JSONL。"
  echo "请先查找：find $DEAR_ROOT/data -type f -iname '*dl19*' -o -iname '*2019*'"
  echo "然后运行：INPUT_FILE=/实际路径/dl19.jsonl bash $0"
  exit 1
fi

if [[ -z "$QRELS_FILE" || ! -s "$QRELS_FILE" ]]; then
  echo "ERROR: 未找到 DL19 qrels。"
  echo "请先查找：find $DEAR_ROOT/data -type f \( -iname '*qrels*19*' -o -iname '*2019qrels*' \)"
  echo "然后运行：QRELS_FILE=/实际路径/qrels.txt bash $0"
  exit 1
fi

RAW_FILE="$RESULT_DIR/dl19_scores.txt"
TREC_FILE="$RESULT_DIR/ranked_dl19.trec"
RUN_LOG="$RESULT_DIR/inference.log"
EVAL_LOG="$RESULT_DIR/evaluation.log"
METADATA="$RESULT_DIR/run_metadata.txt"

{
  echo "run_time=$(date -Iseconds)"
  echo "checkpoint=$CKPT_DIR"
  echo "input_file=$INPUT_FILE"
  echo "qrels_file=$QRELS_FILE"
  echo "gpu_id=$GPU_ID"
  sha256sum "$CKPT_DIR"/adapter_model.*
} | tee "$METADATA"

echo "===== DL19 DeAR original 1k evaluation ====="
echo "CKPT_DIR=$CKPT_DIR"
echo "INPUT_FILE=$INPUT_FILE"
echo "QRELS_FILE=$QRELS_FILE"
echo "RESULT_DIR=$RESULT_DIR"

cd "$CODE_DIR"

if [[ "$FORCE_RERUN" == "1" || ! -s "$RAW_FILE" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" python pointwise_reranker/inference.py \
    --output_dir "$RESULT_DIR/temp" \
    --model_name_or_path "$CKPT_DIR" \
    --tokenizer_name "$STUDENT_MODEL" \
    --teacher_model_name_or_path "$TEACHER_MODEL" \
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
  echo "已有推理结果，跳过 inference：$RAW_FILE"
  echo "如需强制重跑：FORCE_RERUN=1 bash $0"
fi

[[ -s "$RAW_FILE" ]] || { echo "ERROR: 推理结果为空：$RAW_FILE"; exit 1; }

python pointwise_reranker/convert_result_to_trec.py \
  --input "$RAW_FILE" \
  --output "$TREC_FILE"

[[ -s "$TREC_FILE" ]] || { echo "ERROR: TREC 文件为空：$TREC_FILE"; exit 1; }

python -m pyserini.eval.trec_eval \
  -c \
  -m ndcg_cut.10 \
  -m recip_rank \
  -m recall.100 \
  "$QRELS_FILE" \
  "$TREC_FILE" \
  2>&1 | tee "$EVAL_LOG"

echo "===== DL19 evaluation finished ====="
grep -E '^(ndcg_cut_10|recip_rank|recall_100)[[:space:]]+all' "$EVAL_LOG" || true
echo "完整结果：$EVAL_LOG"
