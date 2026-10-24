# MS MARCO 离线难负例挖掘

这一目录完全独立于当前 MVP 训练代码。它只读取模型、checkpoint 和原始数据，最终生成一个新的训练文件；不会修改：

- `pointwise_reranker/`
- `full.sh`
- `/hy-tmp/DeAR/data/MVP-train`
- 原始 `train.jsonl.gz`

最终每条输出固定为 `1 positive + 4 negatives`：优先使用 2 个教师过滤后的 Dense 难负例，再用原始 MS MARCO 负例补足到 4 个。现有训练加载器会完整使用这 4 个负例，无须修改采样或损失代码。

## 文件流程

```text
collection.tsv
  -> prepare_corpus.py       -> corpus.sqlite
  -> encode_corpus.py        -> float16 embedding shards
  -> build_index.py          -> FAISS IVF-PQ index

MS MARCO train.jsonl.gz
  -> retrieve.py             -> Dense Top-100
  -> rerank_student.py       -> Student Top-12
  -> filter_teacher.py       -> teacher-safe mined pool
  -> merge_msmarco.py        -> 2 mined + 2 original
  -> validate_output.py      -> final validation
```

## 运行前说明

先等当前 MVP 训练和 DL19/DL20 评测结束，再运行 GPU 步骤。下面以 10 万 query 的第一轮实验为例；确认效果后，去掉 `retrieve.py` 的 `--limit 100000` 即可跑全量。

预计额外磁盘：collection 压缩包约 2.9GB，解压文本、SQLite、float16 embedding 和索引合计建议预留 30GB 以上。

设置路径：

```bash
export HN_DIR=/hy-tmp/DeAR/hard-negative-mining
export HN_CODE=/root/dear-sll-vv/hard_negative_mining
export HN_PY=/hy-tmp/DeAR/envs/dear-p/bin/python
mkdir -p "$HN_DIR"/{corpus,embeddings,index,intermediate,logs,models}
```

## 1. 下载 MS MARCO passage 全库

Microsoft 官方数据页给出的 passage collection 包含 8,841,823 条记录：

```bash
wget -c \
  https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz \
  -O "$HN_DIR/corpus/collection.tar.gz"

tar -xzf "$HN_DIR/corpus/collection.tar.gz" -C "$HN_DIR/corpus"
test -s "$HN_DIR/corpus/collection.tsv"
```

## 2. 下载统一的 Dense Embedding 模型

这里默认使用英文 E5，并使用其 `query:` / `passage:` 前缀和 mean pooling：

```bash
hf download intfloat/e5-base-v2 \
  --local-dir "$HN_DIR/models/e5-base-v2"
```

Passage 和 query 必须使用同一个模型、pooling 和前缀；这些配置会写入 embedding/index metadata，检索脚本会自动复用。

## 3. 建立 passage 文本库

这是 CPU/磁盘步骤，为后续按 docid 读取候选全文：

```bash
"$HN_PY" "$HN_CODE/prepare_corpus.py" \
  --collection "$HN_DIR/corpus/collection.tsv" \
  --output "$HN_DIR/corpus/corpus.sqlite"
```

## 4. 四卡并行编码全库

每个进程输出独立文件，不共享写句柄：

```bash
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" \
  "$HN_PY" "$HN_CODE/encode_corpus.py" \
    --collection "$HN_DIR/corpus/collection.tsv" \
    --output-dir "$HN_DIR/embeddings" \
    --model "$HN_DIR/models/e5-base-v2" \
    --pooling mean \
    --query-prefix "query: " \
    --passage-prefix "passage: " \
    --num-shards 4 \
    --shard-id "$shard" \
    --batch-size 256 \
    > "$HN_DIR/logs/encode-${shard}.log" 2>&1 &
done
wait
```

如果显存不足，把 `--batch-size 256` 攁为 `128`。

## 5. 建立 FAISS IVF-PQ 索引

```bash
"$HN_PY" "$HN_CODE/build_index.py" \
  --embedding-dir "$HN_DIR/embeddings" \
  --output "$HN_DIR/index/msmarco.ivfpq.faiss" \
  --index-type ivfpq \
  --nlist 16384 \
  --nprobe 32 \
  --pq-m 96 \
  --train-size 1000000 \
  --threads 32
```

## 6. 为训练 query 召回 Dense Top-100

第一轮只挖掘原始 MS MARCO 前 10 万个 query：

```bash
CUDA_VISIBLE_DEVICES=0 \
"$HN_PY" "$HN_CODE/retrieve.py" \
  --train /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --index "$HN_DIR/index/msmarco.ivfpq.faiss" \
  --output "$HN_DIR/intermediate/dense-top100.100k.jsonl.gz" \
  --top-k 100 \
  --nprobe 32 \
  --batch-size 256 \
  --limit 100000
```

这一阶段已经排除所有 `positive_passages` 中出现的 docid。

## 7. 学生选择最难的 Top-12

使用已完成的上一轮学生 checkpoint。不要使用尚未保存完成的目录：

```bash
test -s /hy-tmp/DeAR/outputs/dear-qk-mvp-5pass-seed42-retry/adapter_config.json

CUDA_VISIBLE_DEVICES=0 \
"$HN_PY" "$HN_CODE/rerank_student.py" \
  --input "$HN_DIR/intermediate/dense-top100.100k.jsonl.gz" \
  --corpus-db "$HN_DIR/corpus/corpus.sqlite" \
  --student-model /hy-tmp/DeAR/outputs/dear-qk-mvp-5pass-seed42-retry \
  --output "$HN_DIR/intermediate/student-top12.100k.jsonl.gz" \
  --dense-keep 50 \
  --student-keep 12 \
  --batch-size 16
```

这里还会排除与正例文本完全重复、候选之间文本重复的 passage。

## 8. 教师过滤潜在假负例

学生进程结束后再运行教师，因此一张 GPU 上不会同时加载 3B 和 13B：

```bash
CUDA_VISIBLE_DEVICES=0 \
"$HN_PY" "$HN_CODE/filter_teacher.py" \
  --input "$HN_DIR/intermediate/student-top12.100k.jsonl.gz" \
  --teacher-model /hy-tmp/DeAR/models/llama2-13b-rankllama-teacher \
  --output "$HN_DIR/intermediate/teacher-safe.100k.jsonl.gz" \
  --min-teacher-margin 0.0 \
  --mined-pool-size 8 \
  --batch-size 8
```

过滤条件为：

```text
min(teacher positive scores) - teacher candidate score >= 0
```

教师只负责安全过滤；最终难度顺序仍由学生分数决定。若教师阶段 OOM，将 `--batch-size 8` 攁为 `4`。

## 9. 注入原始 MS MARCO

生成独立的新文件，每条固定 2 个 mined + 2 个 original；某个 query 的安全 mined 不足时自动用 original 补齐：

```bash
"$HN_PY" "$HN_CODE/merge_msmarco.py" \
  --original-train /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --mined "$HN_DIR/intermediate/teacher-safe.100k.jsonl.gz" \
  --output "$HN_DIR/train.hard-mixed.100k.jsonl.gz" \
  --mined-count 2 \
  --total-negatives 4 \
  --only-mined-queries \
  --seed 42
```

## 10. 强制校验

```bash
"$HN_PY" "$HN_CODE/validate_output.py" \
  --input "$HN_DIR/train.hard-mixed.100k.jsonl.gz" \
  --expected-negatives 4
```

校验会检查：每条恰好 4 个负例、正负例 docid 不重叠、文本不重复，并汇总 mined/original 数量。

## 11. 后续单独训练（不要用于当前 MVP 进程）

只有决定开启新的 MS MARCO 难负例实验时，才显式指定这个文件：

```bash
cd /root/dear-sll-vv

TRAIN_DIR="$HN_DIR/train.hard-mixed.100k.jsonl.gz" \
RUN_NAME="dear-qk-msmarco-hardmix-5pass-seed42" \
LAMBDA_RAT=0.02 \
LAMBDA_KK=0 \
LAMBDA_DOWN=0.01 \
TRAIN_N_PASSAGES=5 \
TRAIN_BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=2 \
SEED=42 \
TRAIN_ONLY=0 \
bash full.sh
```

当前 `/hy-tmp/DeAR/data/MVP-train` 不会被上述任何脚本写入或覆盖。

