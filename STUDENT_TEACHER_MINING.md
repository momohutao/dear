# 冻结学生 top1 + 教师过滤

`mine_student_teacher_hard_negatives.py` 默认冻结并复制
`/hy-tmp/DeAR/outputs/dear-kk-gap-lambda-5e-5` 的 adapter/tokenizer，
对原始 `train.jsonl.gz` 全部负例打学生分，取 top1；重新计算教师对全部正例和该 top1 的分数。
仅接受严格 `teacher_positive_score - teacher_negative_score > 0.5` 的正例配对。
top1 未通过不改取 top2；与任何已知正例同 docid/规范化正文的 top1 被排除。
规则标签保留作审计，不参与准入。原训练文件和完整候选池不改动。

这里不要求学生负例分高于正例分；该现象单独记录在 `student_negative_above_positive`。
教师过滤只能降低假负例风险，不能证明真实不相关。

## 挖掘与恢复

```bash
cd /root/dear_kk_mse_new
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
/hy-tmp/DeAR/envs/dear-p/bin/python -u mine_student_teacher_hard_negatives.py \
  --output-dir /hy-tmp/DeAR/data/student_top1_teacher05_7371
```

中断后同命令加 `--resume`。每 128 条保存可恢复进度；恢复会校验输入、权重和实现的 hash。
禁止同时向同一目录运行两个进程。`--limit 128` 可用于独立目录试跑；试跑结果不能用于完整训练。

输出目录：`manifest.json` 记录权重、输入、参数；`audit.jsonl` 保存所有负例学生分及拒绝原因；
`candidates.jsonl` 保存通过的 top1、正例 docid 及教师分；`COMPLETE.json` 标记完整成功。
运行期间只有 `.partial.jsonl` 和 `progress.json`，不要用于训练。

## 后续训练

等全量挖掘成功后：

```bash
cd /root/dear_kk_mse_new
HARD_CANDIDATES=/hy-tmp/DeAR/data/student_top1_teacher05_7371 \
HARD_NEGATIVE_PROBABILITY=0.25 \
DAL_OUTPUT=/hy-tmp/DeAR/outputs/dear-kk-gap-lambda-student-top1-teacher05-p025 \
bash train_rg_dal_1000.sh
```

这条命令按现有脚本从原 base 开始训练，mining student 固定为 73.71 checkpoint。
若明确希望从 73.71 继续训练，另设 `DAL_INIT_CKPT` 为该 checkpoint（脚本续训默认学习率为 `1e-6`）。

Dataset 先照旧随机选正例和 7 篇负例；只有当前正例通过教师过滤时，以 0.25 概率尝试注入 top1。
若 top1 已在随机组内则不替换，否则随机替换一个槽位。其余槽位保留。
因此实际注入率通常低于 25%，日志 `hard_injection_rate` 记录实际替换率。
多正例继续轮换。没有通过过滤的 query 或正例继续使用原始随机采样。
加载时验证完整完成标记、文件 hash、逐行身份与教师 margin，不会把未过滤规则负例当候选。
loss 不变。

## 教师过滤训练与评测（p=0.1）

学生错误门控已从训练代码移除。冻结 73.71 学生仍用于挖掘负例 top1，
但不要求该负例分数高于正例。训练只检查当前正例对应的教师过滤结果：
`teacher_positive_score - teacher_negative_score > 0.5`（若 sidecar 阈值更高则遵循更高阈值）。
已有完整候选目录可直接复用，无需重新挖掘；其中学生分数与错误标记仅用于审计。
不再设置旧的学生门控环境变量或 CLI 参数。

默认注入概率为 `0.1`，默认输出目录后缀为 `p010`。
下面显式使用独立输出目录，避免覆盖旧 p010 实验：

```bash
cd /root/dear_kk_mse_new
HARD_CANDIDATES=/hy-tmp/DeAR/data/student_top1_teacher05_7371 \
DAL_OUTPUT=/hy-tmp/DeAR/outputs/dear-kk-gap-lambda-student-top1-teacher05-p010-teacher-only \
bash train_rg_dal_1000.sh 0.1
# 训练完成后另行执行：
CKPT=/hy-tmp/DeAR/outputs/dear-kk-gap-lambda-student-top1-teacher05-p010-teacher-only \
bash eval_rg_dal_dl19.sh
```

已有外部 `DAL_OUTPUT` / `DAL_INIT_CKPT` 设置仍按原脚本优先级生效。

加载日志 `HARD_CANDIDATES` 报告教师通过配对和 query 数量。
训练日志 `hard_injection_rate` 记录实际替换次数 / 所有采样训练样本数，
同时记录缺少候选、概率未命中、候选已在随机组内、实际替换的次数。
10% 是教师通过后尝试注入的概率，不是全体 query 的替换比例。
正例轮换、原始随机负例采样与 loss 保持不变；随机组已有该负例时不再注入。

冒烟测试：`python audits/teacher_only_hard_negative/smoke.py`（训练环境 Python）。
它验证完整 sidecar 哈希，使用真实训练数据前 128 行、真实 tokenizer、Dataset 和
Collator 做 1280 次采样，不加载模型或执行训练；结果写入同目录。
