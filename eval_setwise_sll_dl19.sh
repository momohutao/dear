#!/usr/bin/env bash
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CODE_DIR=/root/dear_kk_mse_new
CKPT=/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-setwise-sll-new
RESULT_DIR=$CODE_DIR/results/dear-p-3b-sll-kk-setwise-sll-new-ts-contextual-top2-dl19
mkdir -p "$RESULT_DIR"
cd "$CODE_DIR"

python pointwise_reranker/inference.py \
  --output_dir "$RESULT_DIR/temp" \
  --model_name_or_path "$CKPT" \
  --tokenizer_name /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --encode_in_path /hy-tmp/DeAR/data/bm25_beir_dl19_20/dl19.jsonl \
  --encoded_save_path "$RESULT_DIR/dl19.txt" --dataset_name json --loss_type softmax_ce \
  --use_setwise_head true --use_ts_setrank true --ts_judge_type contextual \
  --setwise_feedback_mode topk --setwise_feedback_top_k 2 --setwise_feedback_temperature 2.0 \
  --ts_candidate_topk 100 --ts_budget 5 --ts_batch_size 5 --ts_exploration_rounds 2 \
  --ts_prior_mode dear --ts_prior_strength 4.0 --ts_prior_temperature 1.0 \
  --ts_final_mode fusion --ts_fusion_weight 0.4 --ts_seed 42 \
  --fp16 --per_device_eval_batch_size 32 --q_max_len 32 --p_max_len 196 \
  --dataset_proc_num 1 --dataloader_num_workers 0 --overwrite_output_dir \
  2>&1 | tee "$RESULT_DIR/inference.log"

python pointwise_reranker/convert_result_to_trec.py --input "$RESULT_DIR/dl19.txt" --output "$RESULT_DIR/ranked_dl19.trec"
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 /hy-tmp/DeAR/data/qrels/2019qrels-pass.txt \
  "$RESULT_DIR/ranked_dl19.trec" | tee "$RESULT_DIR/evaluation_results.log"
