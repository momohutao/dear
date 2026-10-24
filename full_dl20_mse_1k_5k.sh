#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1. 环境
# ============================================================

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /hy-tmp/DeAR/envs/dear-p

export DEAR_ROOT=/hy-tmp/DeAR
export HF_HOME=/hy-tmp/DeAR/cache/hf
export HF_DATASETS_CACHE=/hy-tmp/DeAR/cache/datasets
export TMPDIR=/hy-tmp/DeAR/tmp

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================
# 2. 本次实验路径
# ============================================================

export CODE_DIR=/root/dear_mse

export RUN_NAME=dear-p-3b-mse-kk-1w

export CKPT_DIR=/hy-tmp/DeAR/outputs/${RUN_NAME}
export LOG_FILE=/hy-tmp/DeAR/logs/${RUN_NAME}.log
export RESULT_DIR=/root/dear_mse/results/${RUN_NAME}

export BASE_MODEL=/hy-tmp/DeAR/models/Llama-3.2-3B
export TEACHER_MODEL=/hy-tmp/DeAR/models/llama2-13b-rankllama-teacher
export TRAIN_FILE=/hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz

# DL20
export DL20_INPUT=/hy-tmp/DeAR/data/bm25_beir_dl19_20/dl20.jsonl
export DL20_RESULT_ROOT=${RESULT_DIR}/dl20_checkpoints
export DL20_SUMMARY=${DL20_RESULT_ROOT}/checkpoint_scores.tsv

mkdir -p /hy-tmp/DeAR/outputs
mkdir -p /hy-tmp/DeAR/logs
mkdir -p /hy-tmp/DeAR/tmp
mkdir -p "$RESULT_DIR"

cd "$CODE_DIR"

# ============================================================
# 3. 寻找 DL20 qrels
# ============================================================

DL20_QRELS=""

for CANDIDATE in \
  /hy-tmp/DeAR/data/qrels/2020qrels-pass.txt \
  /hy-tmp/DeAR/data/qrels/qrels.dl20-passage.txt \
  /hy-tmp/DeAR/data/qrels/2020qrels-pass.txt.gz
do
  if [ -s "$CANDIDATE" ]; then
    DL20_QRELS="$CANDIDATE"
    break
  fi
done

if [ -z "$DL20_QRELS" ]; then
  echo "错误：没有找到DL20 qrels。"
  echo "请检查以下目录："
  find /hy-tmp/DeAR/data/qrels \
    -maxdepth 1 \
    -type f \
    -printf '%p\n' \
    2>/dev/null || true
  exit 1
fi

echo "DL20 qrels: $DL20_QRELS"

# ============================================================
# 4. 文件预检查
# ============================================================

for REQUIRED_FILE in \
  "$TRAIN_FILE" \
  "$DL20_INPUT" \
  "$DL20_QRELS"
do
  if [ ! -s "$REQUIRED_FILE" ]; then
    echo "错误：文件不存在或为空：$REQUIRED_FILE"
    exit 1
  fi
done

for REQUIRED_DIR in \
  "$BASE_MODEL" \
  "$TEACHER_MODEL"
do
  if [ ! -d "$REQUIRED_DIR" ]; then
    echo "错误：模型目录不存在：$REQUIRED_DIR"
    exit 1
  fi
done

# ============================================================
# 5. CUDA检查
# ============================================================

python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("CUDA版本:", torch.version.cuda)
print("CUDA可用:", torch.cuda.is_available())
print("可见GPU数量:", torch.cuda.device_count())

if not torch.cuda.is_available():
    print("错误：当前环境无法使用CUDA。")
    sys.exit(1)

if torch.cuda.device_count() != 4:
    print(
        "错误：当前脚本需要4张GPU，"
        f"但实际可见{torch.cuda.device_count()}张。"
    )
    sys.exit(1)
PY

# ============================================================
# 6. 检查 Python 代码
# ============================================================

python -m py_compile \
  pointwise_reranker/arguments.py \
  pointwise_reranker/reranker_train.py \
  pointwise_reranker/trainer.py \
  pointwise_reranker/modeling.py \
  pointwise_reranker/region_relation.py \
  pointwise_reranker/inference.py \
  pointwise_reranker/convert_result_to_trec.py

# 确认HfArgumentParser能够识别这个参数。
if ! python pointwise_reranker/reranker_train.py --help \
  2>&1 | grep -q -- "--adapter_save_steps"
then
  echo "错误：arguments.py尚未注册adapter_save_steps。"
  exit 1
fi

# 确认保存Callback已经定义并注册。
if ! grep -q "PeriodicAdapterSaveCallback" \
  pointwise_reranker/trainer.py
then
  echo "错误：trainer.py中没有PeriodicAdapterSaveCallback。"
  exit 1
fi

if ! grep -q "PeriodicAdapterSaveCallback" \
  pointwise_reranker/reranker_train.py
then
  echo "错误：reranker_train.py没有注册保存Callback。"
  exit 1
fi

echo "adapter_save_steps及保存Callback检查正常。"

# ============================================================
# 7. 防止混入旧实验
# ============================================================

if [ -d "$CKPT_DIR" ]; then
  if find "$CKPT_DIR" \
    -maxdepth 2 \
    -type f \
    \( \
      -name "adapter_config.json" \
      -o -name "adapter_model.bin" \
      -o -name "adapter_model.safetensors" \
    \) \
    -print -quit \
    | grep -q .
  then
    echo "错误：输出目录已经存在adapter："
    echo "$CKPT_DIR"
    echo "请修改RUN_NAME，避免覆盖或混合旧实验。"
    exit 1
  fi
fi

mkdir -p "$CKPT_DIR"
mkdir -p "$DL20_RESULT_ROOT"

# ============================================================
# 7. 磁盘预检查
# ============================================================

AVAILABLE_KB=$(
  df --output=avail /hy-tmp |
  tail -n 1 |
  tr -d ' '
)

REQUIRED_KB=$((40 * 1024 * 1024))

if [ "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]; then
  echo "错误：/hy-tmp剩余空间不足40GB。"
  df -h /hy-tmp
  exit 1
fi

echo "磁盘空间检查正常："
df -h /hy-tmp

# ============================================================
# 8. 开始训练
# ============================================================

echo "============================================================"
echo "Stage 1 training starts"
echo "RUN_NAME:  $RUN_NAME"
echo "CKPT_DIR:  $CKPT_DIR"
echo "LOG_FILE:  $LOG_FILE"
echo "DL20_QRELS: $DL20_QRELS"
echo "============================================================"

deepspeed --include localhost:0,1,2,3 \
  pointwise_reranker/reranker_train.py \
  --deepspeed ./config/ds_config.json \
  --output_dir "$CKPT_DIR" \
  --model_name_or_path "$BASE_MODEL" \
  --teacher_model_name_or_path "$TEACHER_MODEL" \
  --temperature 2 \
  --alpha 0.1 \
  --loss_type softmax_ce \
  --lambda_margin_mse 0.01 \
  --use_relation_distill true \
  --lambda_rat 0.02 \
  --lambda_rvr 0 \
  --relation_temperature 2 \
  --relation_heads 8 \
  --student_relation_layer 27 \
  --teacher_relation_layers 39 \
  --relation_use_rope true \
  --relation_eps 1e-8 \
  --dataset_name json \
  --train_dir "$TRAIN_FILE" \
  --fp16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --train_n_passages 2 \
  --learning_rate 1e-4 \
  --q_max_len 32 \
  --p_max_len 196 \
  --max_steps 10000 \
  --logging_steps 20 \
  --save_strategy no \
  --adapter_save_steps 1000 \
  --seed 42 \
  --data_seed 42 \
  --dataset_proc_num 1 \
  --dataloader_num_workers 2 \
  --ddp_timeout 21600 \
  --full_determinism true \
  --report_to none \
  --overwrite_output_dir \
  2>&1 | tee "$LOG_FILE"

echo "===== Training command completed ====="

# ============================================================
# 9. 检查最终adapter
# ============================================================

test -s "$CKPT_DIR/adapter_config.json"

if [ -s "$CKPT_DIR/adapter_model.bin" ]; then
  echo "最终adapter权重：adapter_model.bin"
elif [ -s "$CKPT_DIR/adapter_model.safetensors" ]; then
  echo "最终adapter权重：adapter_model.safetensors"
else
  echo "错误：最终adapter权重不存在。"
  exit 1
fi

# ============================================================
# 10. 检查每1000步adapter
# ============================================================

for STEP in 1000 2000 3000 4000 5000
do
  STEP_CKPT="$CKPT_DIR/adapter-step-$STEP"

  if [ ! -s "$STEP_CKPT/adapter_config.json" ]; then
    echo "错误：缺少checkpoint配置："
    echo "$STEP_CKPT/adapter_config.json"
    exit 1
  fi

  if [ -s "$STEP_CKPT/adapter_model.bin" ]; then
    echo "step $STEP adapter正常：bin"
  elif [ -s "$STEP_CKPT/adapter_model.safetensors" ]; then
    echo "step $STEP adapter正常：safetensors"
  else
    echo "错误：step $STEP缺少adapter权重："
    echo "$STEP_CKPT"
    exit 1
  fi
done

echo "===== All periodic adapters exist ====="

# ============================================================
# 11. 逐个评测DL20
# ============================================================

printf "step\tndcg_cut_10\n" > "$DL20_SUMMARY"

for STEP in 1000 2000 3000 4000 5000
do
  STEP_CKPT="$CKPT_DIR/adapter-step-$STEP"
  STEP_RESULT="$DL20_RESULT_ROOT/step-$STEP"

  mkdir -p "$STEP_RESULT"

  echo "============================================================"
  echo "DL20 inference: step $STEP"
  echo "Adapter: $STEP_CKPT"
  echo "Result:  $STEP_RESULT"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=0 \
  python pointwise_reranker/inference.py \
    --output_dir "$STEP_RESULT/temp" \
    --model_name_or_path "$STEP_CKPT" \
    --tokenizer_name "$BASE_MODEL" \
    --teacher_model_name_or_path "$TEACHER_MODEL" \
    --encode_in_path "$DL20_INPUT" \
    --encoded_save_path "$STEP_RESULT/dl20.txt" \
    --dataset_name json \
    --loss_type softmax_ce \
    --fp16 \
    --per_device_eval_batch_size 32 \
    --q_max_len 32 \
    --p_max_len 196 \
    --dataloader_num_workers 0 \
    --overwrite_output_dir \
    2>&1 | tee "$STEP_RESULT/inference.log"

  test -s "$STEP_RESULT/dl20.txt"

  echo "===== Convert step $STEP to TREC format ====="

  python pointwise_reranker/convert_result_to_trec.py \
    --input "$STEP_RESULT/dl20.txt" \
    --output "$STEP_RESULT/ranked_dl20.trec"

  test -s "$STEP_RESULT/ranked_dl20.trec"

  echo "===== Evaluate DL20 step $STEP ====="

  python -m pyserini.eval.trec_eval \
    -c \
    -m ndcg_cut.10 \
    "$DL20_QRELS" \
    "$STEP_RESULT/ranked_dl20.trec" \
    2>&1 | tee "$STEP_RESULT/evaluation.log"

  SCORE=$(
    awk '
      $1 == "ndcg_cut_10" && $2 == "all" {
        print $3
      }
    ' "$STEP_RESULT/evaluation.log" |
    tail -n 1
  )

  if [ -z "$SCORE" ]; then
    echo "错误：没有提取到step $STEP的DL20 nDCG@10。"
    exit 1
  fi

  printf "%s\t%s\n" \
    "$STEP" \
    "$SCORE" \
    >> "$DL20_SUMMARY"

  echo "step $STEP DL20 nDCG@10 = $SCORE"
done

# ============================================================
# 12. 输出汇总
# ============================================================

echo "============================================================"
echo "DL20 checkpoint results"
echo "============================================================"

if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$DL20_SUMMARY"
else
  cat "$DL20_SUMMARY"
fi

BEST_RESULT=$(
  awk '
    NR > 1 {
      if (!found || $2 > best_score) {
        best_step = $1
        best_score = $2
        found = 1
      }
    }
    END {
      if (found) {
        print best_step, best_score
      }
    }
  ' "$DL20_SUMMARY"
)

BEST_STEP=$(echo "$BEST_RESULT" | awk '{print $1}')
BEST_SCORE=$(echo "$BEST_RESULT" | awk '{print $2}')

echo
echo "最佳step：$BEST_STEP"
echo "最佳DL20 nDCG@10：$BEST_SCORE"
echo "最佳adapter：$CKPT_DIR/adapter-step-$BEST_STEP"
echo "汇总文件：$DL20_SUMMARY"

# ============================================================
# 13. 输出训练危险信号
# ============================================================

echo
echo "===== Recent training metrics ====="

grep -E \
"dear_loss|loss_margin_mse|weighted_margin_mse|loss_rat|weighted_rat|grad_norm|learning_rate" \
"$LOG_FILE" \
| tail -n 20 \
|| true

echo
echo "===== Checkpoint files ====="

find "$CKPT_DIR" \
  -maxdepth 2 \
  -type f \
  \( \
    -name "adapter_config.json" \
    -o -name "adapter_model.bin" \
    -o -name "adapter_model.safetensors" \
  \) \
  -printf '%p\n' \
| sort

echo
echo "===== All finished ====="