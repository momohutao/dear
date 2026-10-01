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

mkdir -p /hy-tmp/DeAR/outputs
mkdir -p /hy-tmp/DeAR/logs

cd /root/DeAR-Reranking

python -m py_compile \
  pointwise_reranker/trainer.py \
  pointwise_reranker/modeling.py \
  pointwise_reranker/region_relation.py

deepspeed --include localhost:0,1,2,3 \
  pointwise_reranker/reranker_train.py \
  --deepspeed ./config/ds_config.json \
  --output_dir /hy-tmp/DeAR/outputs/dear-p-3b-rat-rvr-1k-fixed \
  --model_name_or_path /hy-tmp/DeAR/models/Llama-3.2-3B \
  --teacher_model_name_or_path /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --temperature 2 \
  --alpha 0.1 \
  --loss_type softmax_ce \
  --use_relation_distill true \
  --lambda_rat 0.05 \
  --lambda_rvr 0.05 \
  --relation_temperature 2 \
  --relation_heads 8 \
  --student_relation_layer 27 \
  --teacher_relation_layers 39 \
  --relation_use_rope true \
  --relation_eps 1e-8 \
  --dataset_name json \
  --train_dir /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --fp16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --train_n_passages 2 \
  --learning_rate 1e-4 \
  --q_max_len 32 \
  --p_max_len 196 \
  --max_steps 1000 \
  --logging_steps 10 \
  --save_strategy no \
  --seed 42 \
  --data_seed 42 \
  --dataset_proc_num 1 \
  --dataloader_num_workers 2 \
  --report_to none \
  --overwrite_output_dir \
  2>&1 | tee /hy-tmp/DeAR/logs/dear-p-3b-rat-rvr-1k-fixed.log