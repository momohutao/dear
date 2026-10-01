#!/usr/bin/env bash
# TS-SetRank DL19 inference.  It never retrains or changes the checkpoint.
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CODE_DIR=/root/dear_kk_mse_new
CKPT_DIR=/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new
INPUT_FILE=/hy-tmp/DeAR/data/bm25_beir_dl19_20/dl19.jsonl
QRELS_FILE=/hy-tmp/DeAR/data/qrels/2019qrels-pass.txt

# A real instruction-following causal LM must be supplied.  The repository's
# RankLLaMA teacher is pointwise and must not be used here as a setwise judge.
# JUDGE_MODEL is used only for --setwise_model_name_or_path below.  It never
# changes CKPT_DIR / --model_name_or_path, which remains the DeAR checkpoint.
JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL to a generative setwise judge checkpoint}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ts-dear-fusion-k100-b5-bs10-e2}"
PRIOR_MODE="${PRIOR_MODE:-dear}"
FINAL_MODE="${FINAL_MODE:-fusion}"
FUSION_WEIGHT="${FUSION_WEIGHT:-0.4}"
TS_BUDGET="${TS_BUDGET:-5}"
TS_BATCH_SIZE="${TS_BATCH_SIZE:-10}"
TS_EXPLORATION_ROUNDS="${TS_EXPLORATION_ROUNDS:-2}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-32}"
RESULT_DIR="$CODE_DIR/results/dear-p-3b-sll-kk-new-$EXPERIMENT_NAME-dl19"
mkdir -p "$RESULT_DIR"
cd "$CODE_DIR"

python -m py_compile pointwise_reranker/inference.py pointwise_reranker/ts_setrank.py pointwise_reranker/setwise_judge.py

python pointwise_reranker/inference.py \
  --output_dir "$RESULT_DIR/temp" \
  --model_name_or_path "$CKPT_DIR" \
  --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --encode_in_path "$INPUT_FILE" \
  --encoded_save_path "$RESULT_DIR/dl19.txt" \
  --dataset_name json --loss_type softmax_ce \
  --use_ts_setrank true \
  --ts_candidate_topk 100 --ts_budget "$TS_BUDGET" \
  --ts_batch_size "$TS_BATCH_SIZE" --ts_exploration_rounds "$TS_EXPLORATION_ROUNDS" \
  --ts_prior_mode "$PRIOR_MODE" --ts_prior_strength 4.0 --ts_prior_temperature 1.0 \
  --ts_final_mode "$FINAL_MODE" --ts_fusion_weight "$FUSION_WEIGHT" \
  --ts_seed 42 --ts_judge_type generative \
  --setwise_model_name_or_path "$JUDGE_MODEL" \
  --ts_judge_temperature 0.6 --ts_judge_max_new_tokens "$JUDGE_MAX_NEW_TOKENS" \
  --ts_judge_max_input_tokens 4096 --ts_judge_document_max_chars 1600 \
  --fp16 --per_device_eval_batch_size 32 --q_max_len 32 --p_max_len 196 \
  --dataset_proc_num 1 --dataloader_num_workers 0 --overwrite_output_dir \
  2>&1 | tee "$RESULT_DIR/inference.log"

python pointwise_reranker/convert_result_to_trec.py \
  --input "$RESULT_DIR/dl19.txt" --output "$RESULT_DIR/ranked_dl19.trec"
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 "$QRELS_FILE" \
  "$RESULT_DIR/ranked_dl19.trec" | tee "$RESULT_DIR/evaluation_results.log"
