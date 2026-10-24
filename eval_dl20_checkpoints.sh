#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export CODE_DIR=/root/dear74.45+beir+dl20
export BASE_MODEL=/hy-tmp/DeAR/models/Llama-3.2-3B
export TEACHER_MODEL=/hy-tmp/DeAR/models/llama2-13b-rankllama-teacher

export CKPT_DIR=/hy-tmp/DeAR/outputs/dear+kk+mse5000
export INPUT_FILE=/hy-tmp/DeAR/data/bm25_beir_dl19_20/dl20.jsonl
export RESULT_ROOT=/root/dear74.45/results/dear+kk+mse5000/dl20_checkpoints

# 自动寻找DL20 qrels。
QRELS_FILE=""

for candidate in \
  /hy-tmp/DeAR/data/qrels/2020qrels-pass.txt \
  /hy-tmp/DeAR/data/qrels/qrels.dl20-passage.txt
do
  if [ -s "$candidate" ]; then
    QRELS_FILE="$candidate"
    break
  fi
done

if [ -z "$QRELS_FILE" ]; then
  echo "没有找到DL20 qrels。请先执行："
  echo "find /hy-tmp/DeAR/data -type f -iname '*2020*qrel*' -o -iname '*qrel*dl20*'"
  exit 1
fi

test -s "$INPUT_FILE"

mkdir -p "$RESULT_ROOT"
cd "$CODE_DIR"

SUMMARY_FILE="$RESULT_ROOT/checkpoint_scores.tsv"
printf "step\tndcg_cut_10\n" > "$SUMMARY_FILE"

for step in 1000 2000 3000 4000 5000
do
  ADAPTER_DIR="$CKPT_DIR/adapter-step-$step"
  STEP_DIR="$RESULT_ROOT/step-$step"

  test -s "$ADAPTER_DIR/adapter_config.json"
  test -s "$ADAPTER_DIR/adapter_model.bin"

  mkdir -p "$STEP_DIR"

  echo "===== DL20 inference: step $step ====="

  CUDA_VISIBLE_DEVICES=0 \
  python pointwise_reranker/inference.py \
    --output_dir "$STEP_DIR/temp" \
    --model_name_or_path "$ADAPTER_DIR" \
    --tokenizer_name "$BASE_MODEL" \
    --teacher_model_name_or_path "$TEACHER_MODEL" \
    --encode_in_path "$INPUT_FILE" \
    --encoded_save_path "$STEP_DIR/dl20.txt" \
    --dataset_name json \
    --loss_type softmax_ce \
    --fp16 \
    --per_device_eval_batch_size 32 \
    --q_max_len 32 \
    --p_max_len 196 \
    --dataloader_num_workers 0 \
    --overwrite_output_dir

  test -s "$STEP_DIR/dl20.txt"

  python pointwise_reranker/convert_result_to_trec.py \
    --input "$STEP_DIR/dl20.txt" \
    --output "$STEP_DIR/ranked_dl20.trec"

  python -m pyserini.eval.trec_eval \
    -c \
    -m ndcg_cut.10 \
    "$QRELS_FILE" \
    "$STEP_DIR/ranked_dl20.trec" \
    2>&1 | tee "$STEP_DIR/evaluation.log"

  SCORE=$(
    awk '
      $1 == "ndcg_cut_10" && $2 == "all" {
        print $3
      }
    ' "$STEP_DIR/evaluation.log" |
    tail -n 1
  )

  if [ -z "$SCORE" ]; then
    echo "step $step 未提取到nDCG@10"
    exit 1
  fi

  printf "%s\t%s\n" \
    "$step" \
    "$SCORE" \
    >> "$SUMMARY_FILE"
done

echo "===== DL20 checkpoint summary ====="
column -t -s $'\t' "$SUMMARY_FILE"