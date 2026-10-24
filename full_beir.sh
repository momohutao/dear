#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export DEAR_ROOT="${DEAR_ROOT:-/hy-tmp/DeAR}"
export CODE_DIR="${CODE_DIR:-/root/dear-sll-vv}"
# export CKPT_DIR=/hy-tmp/DeAR/outputs/dear-p-3b-rat-rvr-1k-l015
# export BM25_DIR=/hy-tmp/DeAR/data/bm25_beir_dl19_20
# export QRELS_DIR=/hy-tmp/DeAR/data/qrels/beir-v1.0.0
# export RESULT_ROOT=/root/dear74.45/results/beir8_l015
export CKPT_DIR="${CKPT_DIR:-/hy-tmp/DeAR/outputs/dear-p-3b-original-1k-fixed}"
export BM25_DIR="${BM25_DIR:-/hy-tmp/DeAR/data/bm25_beir_dl19_20}"
export QRELS_DIR="${QRELS_DIR:-/hy-tmp/DeAR/data/qrels/beir-v1.0.0}"
export RESULT_ROOT="${RESULT_ROOT:-/root/dear-sll-vv/results/dear_beir}"

export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

mkdir -p \
  "$RESULT_ROOT" \
  "$QRELS_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TMPDIR"

cd "$CODE_DIR"

test -s "$CKPT_DIR/adapter_config.json"
test -s "$CKPT_DIR/adapter_model.bin"

test -s \
  /hy-tmp/DeAR/models/Llama-3.2-3B/config.json

test -s \
  /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher/config.json

DATASETS=(
  covid
  nfc
  touche
  dbpedia
  scifact
  signal
  news
  robust04
)

declare -A QRELS=(
  [covid]="$QRELS_DIR/qrels.beir-v1.0.0-trec-covid.test.txt"
  [nfc]="$QRELS_DIR/qrels.beir-v1.0.0-nfcorpus.test.txt"
  [touche]="$QRELS_DIR/qrels.beir-v1.0.0-webis-touche2020.test.txt"
  [dbpedia]="$QRELS_DIR/qrels.beir-v1.0.0-dbpedia-entity.test.txt"
  [scifact]="$QRELS_DIR/qrels.beir-v1.0.0-scifact.test.txt"
  [signal]="$QRELS_DIR/qrels.beir-v1.0.0-signal1m.test.txt"
  [news]="$QRELS_DIR/qrels.beir-v1.0.0-trec-news.test.txt"
  [robust04]="$QRELS_DIR/qrels.beir-v1.0.0-robust04.test.txt"
)

echo "===== Checking local BEIR qrels ====="

for dataset in "${DATASETS[@]}"; do
  qrels_file="${QRELS[$dataset]}"

  if [[ ! -s "$qrels_file" ]]; then
    echo "ERROR: missing qrels for $dataset"
    echo "Expected file: $qrels_file"
    exit 1
  fi

  echo "[$dataset] qrels OK: $qrels_file"
done

run_dataset() {
  local dataset="$1"
  local gpu_id="$2"

  local input_file="$BM25_DIR/${dataset}.jsonl"
  local result_dir="$RESULT_ROOT/${dataset}"
  local raw_file="$result_dir/${dataset}.txt"
  local trec_file="$result_dir/ranked_${dataset}.trec"
  local eval_file="$result_dir/evaluation.log"
  local run_log="$result_dir/run.log"

  mkdir -p "$result_dir"

  if [[ ! -s "$input_file" ]]; then
    echo "ERROR: missing $input_file"
    return 1
  fi

  echo "[$dataset] starting on physical GPU $gpu_id"

  if [[ ! -s "$raw_file" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" python \
      pointwise_reranker/inference.py \
      --output_dir "$result_dir/temp" \
      --model_name_or_path "$CKPT_DIR" \
      --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
      --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
      --encode_in_path "$input_file" \
      --encoded_save_path "$raw_file" \
      --dataset_name json \
      --loss_type softmax_ce \
      --fp16 \
      --per_device_eval_batch_size 32 \
      --q_max_len 32 \
      --p_max_len 196 \
      --dataset_proc_num 2 \
      --dataloader_num_workers 2 \
      --overwrite_output_dir \
      2>&1 | tee "$run_log"
  else
    echo "[$dataset] raw inference result exists; skipping inference"
  fi

  test -s "$raw_file"

  python \
    pointwise_reranker/convert_result_to_trec.py \
    --input "$raw_file" \
    --output "$trec_file"

  test -s "$trec_file"

  python -m pyserini.eval.trec_eval \
    -c \
    -m ndcg_cut.10 \
    "${QRELS[$dataset]}" \
    "$trec_file" \
    2>&1 | tee "$eval_file"

  echo "[$dataset] finished"
}

if [[ -n "${BEIR_DATASET:-}" ]]; then
  run_dataset "$BEIR_DATASET" "${BEIR_GPU:-0}"
  exit 0
fi

PIDS=()

for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"
  gpu_id=$((index % 4))

  run_dataset "$dataset" "$gpu_id" &
  PIDS+=("$!")

  if (( ${#PIDS[@]} == 4 )); then
    for pid in "${PIDS[@]}"; do
      wait "$pid"
    done

    PIDS=()
  fi
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

SUMMARY="$RESULT_ROOT/beir8_summary.tsv"

printf "dataset\tndcg@10\tndcg@10_x100\n" > "$SUMMARY"

for dataset in "${DATASETS[@]}"; do
  eval_file="$RESULT_ROOT/$dataset/evaluation.log"

  score=$(
    awk '
      $1 == "ndcg_cut_10" && $2 == "all" {
        print $3
      }
    ' "$eval_file" | tail -1
  )

  if [[ -z "$score" ]]; then
    score="NA"
    score_x100="NA"
  else
    score_x100=$(
      awk -v value="$score" '
        BEGIN {
          printf "%.2f", value * 100
        }
      '
    )
  fi

  printf "%s\t%s\t%s\n" \
    "$dataset" \
    "$score" \
    "$score_x100" \
    >> "$SUMMARY"
done

awk '
  BEGIN {
    FS = "\t"
  }

  NR > 1 && $2 != "NA" {
    sum += $2
    count += 1
  }

  END {
    if (count > 0) {
      printf "BEIR-8 mean nDCG@10: %.4f\n", sum / count
      printf "BEIR-8 mean nDCG@10 x100: %.2f\n", 100 * sum / count
    }
  }
' "$SUMMARY" | tee "$RESULT_ROOT/beir8_mean.log"

echo "===== BEIR-8 summary ====="
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$SUMMARY"
else
  cat "$SUMMARY"
fi
cat "$RESULT_ROOT/beir8_mean.log"
