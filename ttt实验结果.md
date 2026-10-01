# DeAR 查询级测试时适配（TTA/TTT）实验结果

## 实验目标与固定设置

目标是在 DeAR pointwise reranker 的推理阶段，为每条 query 临时优化低秩矩阵 `A_q`、`B_q`，以残差形式修正 DeAR 分数；处理下一条 query 时重新初始化矩阵。

固定实验条件：

- 数据集：TREC DL19 passage。
- checkpoint：`/hy-tmp/DeAR/outputs/dear-p-3b-kk-new`。
- 指标：`nDCG@10`。
- 候选：每条 query 的前 100 个候选。
- 原始伪标签：前 5 名伪正例、81--100 名伪负例。
- 低秩维度：除特别说明外为 `rank=4`。
- 初始矩阵：`A ~ N(0, 1e-3)`，`B=0`。
- DeAR、teacher 均被冻结；梯度仅更新当前 query 的 `A_q`、`B_q`。

## 实现改动

已完成下列推理期实现：

1. 初版低秩实现利用已有 `region_ids` 从 student 最后一层 hidden states 分别 mean-pool query 区域与 document 区域，并做 L2 归一化。
2. 在 query 内标准化 DeAR logit：`z_hat=(z-mean(z))/(std(z)+eps)`。
3. 残差打分使用：

   ```text
   s = z_hat + gamma * tanh(q_hat^T A_q B_q^T d_hat)
   ```

4. 使用加权伪正/伪负 pairwise logistic loss。
5. 加入残差锚定正则：

   ```text
   loss = pair_loss + anchor_weight * mean(residual^2)
   ```

6. 可选加载 RankLLaMA teacher；teacher 仅用于筛选伪标签，不参与最终分数融合。
7. `full.sh` 支持用环境变量覆盖 TTA 超参数，例如：

   ```bash
   TTA_GAMMA=0.2 TTA_STEPS=5 TTA_ANCHOR_WEIGHT=0.1 bash full.sh
   ```

8. 支持基于原始 DeAR 排名设置逐文档残差系数：`tta_protect_top_k` 将前若干名的残差设为 0，`tta_top10_gamma` 可单独设置保护边界到 rank 10 之间的残差系数。
9. 当前适配表示已改为 DeAR 分类头真正读取的最后有效 token 表示。对每个候选提取并归一化 `h_hat`，每条 query 只优化一个初始为零的 `u_q ∈ R^3072`：

   ```text
   s = z_hat + gamma * tanh(h_hat^T u_q)
   ```

   该形式初始残差严格为零，且 `u_q=0` 时仍有非零梯度；不再依赖低秩矩阵的双因子优化。
10. 支持 teacher 驱动的 Top-30 排序纠错目标：候选集合为 `DeAR rank 6--30 ∪ teacher top-5`；仅使用 teacher 与 DeAR 偏好方向相反且 teacher 分差超过 `delta_T` 的文档对。每对的 teacher 偏好为监督方向，权重为 `min(1, |Δt| / c)`。

## DL19 结果汇总

| 实验 | rank | gamma | steps | lr | anchor | teacher 筛选 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---|---:|
| 原始 DeAR（无 TTA） | - | - | - | - | - | 否 | 0.7350 |
| 强 TTA（早期版本） | 8 | 1.0 | 20 | 1e-2 | 0 | 否 | 0.7143 |
| 保守 TTA | 4 | 0.05 | 3 | 1e-3 | 1.0 | 否 | 0.7353 |
| 保守 TTA | 4 | 0.10 | 5 | 1e-3 | 0.1 | 否 | 0.7353 |
| 保守 TTA | 4 | 0.10 | 5 | 1e-3 | 0.3 | 否 | 0.7353 |
| 保守 TTA | 4 | 0.20 | 5 | 1e-3 | 0.1 | 否 | 0.7353 |
| teacher 一致性伪标签 + 保守 TTA | 4 | 0.10 | 5 | 1e-3 | 0.1 | 是 | 0.7353 |
| top-3 硬保护：rank 4--100 适配 | 4 | 0.10 | 5 | 1e-3 | 0.1 | 否 | 0.7353 |
| 分段保护：top-3=0、rank 4--10=0.01、rank 11--100=0.05 | 4 | 分段 | 5 | 1e-3 | 0.1 | 否 | 0.7353 |
| 分类位置向量 `u_q`（3072 维） | - | 0.10 | 5 | 1e-3 | 0.1 | 否 | 0.7350 |
| 分类位置向量 + teacher Top-30 纠错 pairs | - | 0.10 | 5 | 1e-3 | 0.1 | 是 | 0.7353 |
| SLL+KK checkpoint + 分类位置向量 teacher Top-30 纠错 | - | 0.10 | 5 | 1e-3 | 0.1 | 是 | **0.7419** |

## 关键观察

### 1. 强 TTA 会明显破坏原始排序

`gamma=1.0, steps=20, lr=1e-2, rank=8` 将结果从 `0.7350` 降至 `0.7143`。

原因如下：

- 未限制的双线性残差出现了远大于 `z_hat` 的数值，曾观察到约 `33.96` 和 `-14.28` 的输出；原始标准化分数的尺度约为 1。
- pairwise loss 只要求伪正例高于伪负例，未约束 top-10 内部的细粒度次序。
- 第 6--80 名候选不参与伪标签 loss，但仍会被低秩矩阵外推打分，可能被错误推入 top-10。
- 强 TTA 与原排序相比，43/43 个 query 的 top-10 顺序发生变化，平均 top-1 文档重合率仅为 0.581。

### 2. 有界残差和锚定正则消除了显著退化

加入 `tanh` 并降低 `gamma`、学习率和步数后，TTA 不再压过 DeAR 原始分数。保守组相对无 TTA 的结果为 `+0.0003`，但该幅度很小，暂不能视为明确有效提升。

`gamma=0.2` 的实验中，仅 2/43 个 query 的 top-10 内部次序变化，且 top-10 候选集合没有变化。

### 3. anchor=0.1 和 anchor=0.3 暂无可观察差异

在 `gamma=0.1, steps=5, lr=1e-3, rank=4` 下，两组均得到 `0.7353`。两组仅 1/43 个 query 的完整候选排序不同，top-10 完全相同；说明当前残差幅度仍很小。

### 4. RankLLaMA teacher 一致性筛选已接入，但尚未改变最终排名

teacher 筛选规则为：

```text
伪正例 = DeAR top-5 ∩ teacher top-10
伪负例 = DeAR 81--100 ∩ teacher 81--100
若任一交集为空，则跳过该 query 的 TTA。
```

在 `gamma=0.1, steps=5, lr=1e-3, rank=4, anchor=0.1` 下，teacher 版本的结果仍为 `0.7353`，与无 teacher 的保守组最终排序相同。该 teacher 路径已实际加载并执行 13B RankLLaMA 前向；当前无差异表明受限残差尚不足以改变排序边界，而不是 teacher 未被调用。

### 5. 头部保护策略实验

为保护 nDCG@10 最敏感的位置，代码根据**初始 DeAR 排名**为每个候选设置不同的残差系数。两组实验均关闭 teacher，以隔离头部保护策略本身的影响，并固定 `rank=4, steps=5, lr=1e-3, anchor=0.1`。

1. 硬保护：原始 rank 1--3 的残差系数为 0，rank 4--100 使用 `gamma=0.10`。
2. 平滑分段保护：原始 rank 1--3 的残差系数为 0；rank 4--10 为 `0.01`；rank 11--100 为 `0.05`。

两组均得到 `nDCG@10=0.7353`。与无 TTA 的 DeAR 排序相比，43/43 个 query 的 top-3 和 top-10 排序均保持不变。这证明保护逻辑正确阻止了头部文档被残差扰动；但在当前保守优化强度下，也尚未产生可观察的排名提升。

### 6. 直接适配分类位置表示 `u_q`

将 Q/D mean-pooling 低秩残差替换为分类头实际使用的最后有效 token 表示 `h_hat_i`，每条 query 仅优化一个 3072 维、初始为零的向量 `u_q`：

```text
s_i = z_hat_i + gamma * tanh(h_hat_i^T u_q)
```

本轮参数为 `gamma=0.10, steps=5, lr=1e-3, anchor=0.1`，关闭 teacher 和头部保护。DL19 得到 `nDCG@10=0.7350`，与无 TTA 基线相同，低于低秩保守版的 `0.7353`。

该版本不是未生效：相对原始 DeAR，21/43 个 query 的完整候选排序发生变化，2/43 个 query 的 top-10 内部顺序发生变化；只是这些变化没有带来 nDCG@10 的净提升。该结果表明直接分类位置表示在当前伪标签与保守优化参数下可行且稳定，但尚未优于原始 DeAR。

### 7. teacher 驱动的 Top-30 边界纠错

在分类位置向量 `u_q` 的基础上，将原先“DeAR top-5 高于 rank 81--100”的自确认式伪标签，替换为 teacher 对边界顺序的纠错监督。

纠错候选集合与 loss 为：

```text
候选集合 = DeAR rank 6--30 ∪ teacher top-5
C = {(i, j): sign(z_i-z_j) != sign(t_i-t_j), |t_i-t_j| > delta_T}
y_ij = sign(t_i-t_j)
w_ij = min(1, |t_i-t_j| / c)
L_corr = sum[w_ij * softplus(-y_ij * (s_i-s_j))] / sum[w_ij]
```

本轮配置：`gamma=0.10, steps=5, lr=1e-3, anchor=0.1, delta_T=0.1, c=1.0`；启用 13B RankLLaMA teacher，仅用于构造纠错 pairs，不参与最终分数融合。

结果为 `nDCG@10=0.7353`，较无 TTA 基线 `0.7350` 增加 `0.0003`，也优于同参数、但不使用纠错 loss 的分类位置向量版本（`0.7350`）。相对原始 DeAR，25/43 个 query 的完整排序改变；但 43/43 个 query 的 top-10 排序与 top-10 候选集合均未变化。说明该保守配置的纠错主要作用于 top-10 以下，成功避免了头部退化，但提升幅度仍很小。

### 8. SLL+KK checkpoint 上复用最佳 TTT

在新训练完成的 checkpoint `/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new` 上，复用此前表现最好的 TTT 配置：分类位置向量 `u_q`、teacher Top-30 纠错 pairs、`gamma=0.10`、`steps=5`、`lr=1e-3`、`anchor=0.1`、`delta_T=0.1`、`c=1.0`。student 与 13B teacher 均冻结；teacher 仅用于构造纠错 pair，不参与最终分数融合。

DL19 的评测结果为 **`nDCG@10=0.7419`**。这比旧 checkpoint 上同一 TTT 配置的 `0.7353` 高 `0.0066`。

注意：尚未在这个新 SLL+KK checkpoint 上单独运行“无 TTT”的 DL19 基线，因此目前只能得出“新 checkpoint + 此 TTT 配置达到 0.7419”，不能将提升完全归因于 TTT；它也可能主要来自新训练得到的 checkpoint。

## 当前结论

当前最安全配置是有界残差加锚定正则。它避免了原始强 TTA 的大幅退化，但还没有获得显著提升。

仅依赖 DeAR 自己的 top-5/top-100 后段生成伪标签本质上是自训练，可能放大初始排序错误。后续优先方向应是提高伪标签质量，而不是继续无约束增大 `gamma` 或优化步数：

1. 在独立开发集而不是 DL19 上继续调参，DL19 只保留为最终报告集。
2. 统计每条 query 的 teacher 一致伪正/伪负例数量，按一致性强度决定是否适配。
3. 结合 BM25 与 teacher，采用三方一致的伪标签，或仅对高置信 query 启动 TTA。
4. 保留 `tanh`、锚定正则和 query 级重新初始化；不要恢复 `gamma=1, steps=20, lr=1e-2` 的强适配设置。

## TS-SetRank（推理后处理，2026-09-26）

本轮新增的是独立于已有 TTA/TTT 的 `TS-SetRank` 推理分支；**没有改训练 loss、训练数据、DeAR 分类头，也没有覆盖已有 TTA 实现**。固定 checkpoint 为 `/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new`，候选为 DL19 每个 query 的 100 篇 BM25 候选。

实现内容：

1. 每个 query 在 DeAR 初始排序之后新建 100 个 Beta 后验；query 结束立即丢弃。支持 `uniform` 和 `dear` 两种先验。后者实现为 `alpha=1+4*sigmoid(z_hat/1.0)`、`beta=1+4*(1-sigmoid(z_hat/1.0))`。
2. 前若干轮优先从未观测文档均匀采样，后续用独立的 `random.Random(ts_seed + query_index)` 作 Thompson Beta 采样；同一轮无重复、不同轮可重复。
3. 增加 `SetwiseJudge` 抽象接口、`MockSetwiseJudge` 和 `GenerativeSetwiseJudge`。生成式 judge 在**同一 prompt**中传入 query 及整批 `Passage ID P0: ...` 文档；主反馈协议为纯 ID 集合（如 `P0 P3 P4`），用 `re.findall(r"\bP\d+\b", first_answer_line)` 映射回真实 docid，同时保留旧 JSON 的兼容解析。无可用 ID 时会用更严格提示重试一次，二次失败返回无反馈而不更新后验。
4. 为避免 tokenizer 的右侧截断悄悄丢掉批次末尾文档，生成式 prompt 会按所有文档共同的字符上限缩短，直到整批临时 ID 都在模型上下文内；不会出现“传入 P0--P9、实际只看到 P0--P7”的错位。
5. 支持 `posterior` 和 `fusion` 两种最终分数；fusion 使用 DeAR score z-score 与 posterior-logit z-score 的安全融合。仅重排 `ts_candidate_topk`，范围外文档按原始 DeAR 顺序追加，且写出严格递减的 TREC score，避免评测器把范围外文档重新插回 top-K。
6. 每个 query 都输出可审计 trace：`selection_count`、`sample_count`、`success_count`、`failure_count`、`alpha`、`beta`、`posterior_mean`、初始 rank/score；默认文件为 `<dl19.txt>.ts_setrank_trace.jsonl`。

### 审查与验证

- `python -m py_compile` 通过：`arguments.py`、`data.py`、`inference.py`、`setwise_judge.py`、`ts_setrank.py`。
- Mock 单元测试通过：固定 seed 的两次运行完全一致；top-K 外文档没有进入重排区；4 轮、每轮 2 篇的有效反馈共留下 8 次后验观测；无反馈时不更新任一后验。
- DL19 端到端控制运行通过：43 个 query、每个 100 篇文档，trace 完整生成；Mock 默认不伪造相关性也不读取 qrels，因此 43 个 query 均为 5 次“无有效反馈”、共记录 2,150 次批次选择、总后验更新为 0。输出的 43 个 query 的 100 篇 docid 顺序均与原始 DeAR 完全一致，证明 Top-K 边界和无反馈保护正确。
- `use_ts_setrank=false` 时不会构造 judge、不会执行 TS 采样；DeAR 模型输入没有变更。为 setwise prompt 新增的 raw query/document 仅是推理元数据，不进入 student/teacher 的 token ids。

### 结果

| 实验 | prior / final | budget / batch / explore | judge | nDCG@10 | 说明 |
|---|---|---:|---|---:|---|
| TS-SetRank Mock 无反馈控制 | dear / fusion(0.4) | 5 / 10 / 2 | Mock（不提供标签） | 0.7414 | 与当前 input-order DeAR 排序完全一致；仅验证集成，不是有效的 setwise 相关性实验 |
| TS-SetRank + Llama-3.2-3B 基座 judge | dear / fusion(0.4) | 5 / 10 / 2 | 生成式 CausalLM，纯 P-ID 协议 | **0.7376** | 真正完成 43 个 query 的 setwise 反馈与后验更新；相对 0.7414 为 -0.0038 |

此前保存的 SLL+KK 原始结果为 `0.7411`（常规 score 排序）和 `0.7414`（input-order/tie-safe 排序）。本轮用户要求不重复原始 checkpoint 的完整基线；控制实验的 `0.7414` 是 TS 分支无反馈时保持同一排序得到的回归结果，不应表述为 TS 带来的提升。

纯 ID 协议修正了基座 Llama 的 JSON/空输出问题：`Passage ID P0:` 标签配合最少 4 个生成 token 后，它可以输出可解析的 `P0 P3` 类标识符；解析只取第一个非空回答行，避免将后续解释中出现的 passage ID 误当正例。DL19 全量运行中 43/43 个 query 都有有效后验更新，`feedback_failures=0`，共 2,150 次选择和 2,150 次后验观测。

不过该基座模型的**相关性判断质量很差**：2,127/2,150（98.93%）观测被判为正例，仅 23 次负例；它常把当前批次的大部分甚至全部 `P0...P9` 都复制回来。这解释了 `0.7376` 的退化：格式和 TS 采样正确，但反馈几乎没有区分相关与不相关文档。RankLLaMA 仍是 pointwise sequence classifier，不能替代 setwise judge。因此这是一条有效的 TS-SetRank 端到端实验，但不能作为有效的提升结果。获得更有区分力的 instruction/setwise checkpoint 后，可用：

```bash
JUDGE_MODEL=/path/to/instruction-setwise-model \\
EXPERIMENT_NAME=ts-dear-fusion-k100-b5-bs10-e2 \\
bash /root/dear_kk_mse_new/run_ts_setrank_dl19.sh
```

可通过环境变量切换两种要求的实验：`PRIOR_MODE=uniform|dear`、`FINAL_MODE=posterior|fusion`、`FUSION_WEIGHT=0.4`、`TS_BUDGET`、`TS_BATCH_SIZE`、`TS_EXPLORATION_ROUNDS`。脚本固定使用 `dear-p-3b-sll-kk-new`，不会训练或覆盖现有 TTA 结果。

### 两阶段 DeAR / Instruct judge 改造（待下载后评测）

按最新要求，TS-SetRank 现使用两个完全独立的参数：

```text
--model_name_or_path             /hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new
--setwise_model_name_or_path     /hy-tmp/DeAR/models/Llama-3.2-3B-Instruct
```

其中前者只加载冻结的 DeAR pointwise checkpoint；后者只加载 `GenerativeSetwiseJudge`。`JUDGE_MODEL` 环境变量在 `run_ts_setrank_dl19.sh` 中只写入 `--setwise_model_name_or_path`，不能覆盖 DeAR 的 `CKPT_DIR`。

推理相位已改为：

```text
DeAR GPU pointwise scoring
→ all_results（CPU float/string 原始分数与文本缓存）
→ del DeAR + gc.collect + torch.cuda.empty_cache
→ 加载独立 Instruct setwise judge
→ TS posterior update 与最终融合（CPU）
```

已通过 Python 编译、参数检查以及 DL19 两阶段 Mock smoke test：日志明确先输出 `Released frozen DeAR pointwise model`，之后才构造独立 judge；43 个 query 的原始 DeAR 分数已经 CPU 缓存并完成 TS 输出。2026-09-26 下载 `meta-llama/Llama-3.2-3B-Instruct` 时，环境返回 `GatedRepoError: 401 Unauthorized`：当前 Hugging Face 未登录，且没有该 gated Meta 仓库访问权限。只下载到 README/LICENSE，未获得 config、tokenizer 或权重，因此**尚未使用该目标 Instruct 模型运行 DL19，也没有新的 nDCG 结果**。不能用现有 Llama-3.2-3B 基座或 RankLLaMA 替代这一指定实验。

随后也测试了 `HF_ENDPOINT=https://hf-mirror.com`。该镜像对 `config.json` 返回 `308` 到官方 Hugging Face，官方仍返回 `GatedRepo 401`；所以镜像只能改善网络连通性，不能绕过 Meta Llama 的账户授权。

取得并授权一个已接受 Llama 3.2 license 的 HF token 后，执行：

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \\
  --local-dir /hy-tmp/DeAR/models/Llama-3.2-3B-Instruct

JUDGE_MODEL=/hy-tmp/DeAR/models/Llama-3.2-3B-Instruct \\
bash /root/dear_kk_mse_new/run_ts_setrank_dl19.sh
```

## RankLLaMA 连续分布反馈的查询级 TTA（2026-09-26）

本轮新增一个与旧的 pairwise TTA、TOUR-list reward、TS-SetRank 和 setwise residual **互斥且独立**的纯推理分支；没有改动训练目标、训练数据或 checkpoint。固定学生 checkpoint 为：

```text
/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new
```

### 实现与审查

- 入口是 `pointwise_reranker/inference.py` 的 `--use_distribution_tta`。旧 `adapt_one_query()`（pairwise）和 `adapt_one_query_listwise_reward()` 均保持不变；多个后处理器同时启用会报错。
- DeAR 使用原有 `model_output.scores` 产生 pointwise logit；缓存的是 `model_output.setwise_embeddings`，即 classification head 实际读取的最后有效 token 隐藏状态。`modeling.py` 已逐样本校验该位置与唯一的 `region_ids==2` 一致，因而没有误用 Q/D mean pooling 或错误 token。
- 先对每个 query 的 100 篇候选运行冻结 DeAR，得到并 CPU 缓存 `z_i,h_i`；随后释放 DeAR GPU 显存，RankLLaMA 只对原始 Top-30 评分。两者均 `detach()`；适配阶段唯一 `requires_grad=True` 的对象是每条 query 新建、FP32、零初始化的 `u_q`，处理下一条 query 时立即丢弃。
- 分数为 `s_i=z_hat_i+gate_q*0.1*tanh(h_hat_i^T u_q)`。总 loss 为全局 `KL(P_T||P_S)` 加 5 个固定滑窗 `(1--10,6--15,11--20,16--25,21--30)` 的位置折扣局部 KL、`0.1*mean(residual^2)` 和 `1e-4*mean(u_q^2)`；Top-31--100 始终按原始 DeAR 顺序追加并用严格递减 TREC score 写出，不能跨入 Top-30。
- 教师缓存以 `(qid, docid, teacher checkpoint absolute path)` 为键写入 JSONL，而不是按候选位置缓存；因此候选重排不会造成 teacher-score 对错文档。正式缓存：`/hy-tmp/DeAR/cache/teacher_scores/rankllama_dl19_distribution_tta_v1.jsonl`。
- 审查时修复了两个优化细节：早停改为比较**更新后**目标，并保存任一严格更优的 `u_q`；当 Adam 最大步长使完整目标上升时，仅对 `u_q` 作回溯，并退回真实负梯度方向。这样不会因小尺度 KL 的 `1e-5` 容差把有效更新误丢弃，也不会接受增大目标的更新。

### 验证

- `py_compile`、`bash -n full.sh run_distribution_tta_dl19.sh` 通过；合成单元检查通过：`gamma=0`/`steps=0` 保持原序、每条 query 都重新创建 `u_q`、模型参数无梯度、KL 下降、Top-K 外不变、缓存 qid/docid 对齐、30 篇产生完整 5 个滑窗，小于 30 篇动态缩窗。
- 2-query 冒烟运行完成：首次缓存写入 60 条；复跑为 `60` 次命中、`0` 次 miss。没有 NaN；最终 KL 不高于初始 KL，适配排序仅发生在 Top-30 内。
- 正式 DL19 的原始 pointwise 对照重新运行得到 `0.7411`，与指定基线一致，证明关闭新开关不会改变 DeAR。

### DL19 全量消融

所有 TTA 运行均为 `K=30, gamma=0.10, steps=10, lr=1e-2, T_teacher=T_student=2.0, anchor=0.1, u_norm=1e-4`。43 个 query 都完成，无非有限值回退。

| 实验 | 局部 KL | 熵 gate | nDCG@10 | 相对 raw |
|---|---:|---:|---:|---:|
| A. 原始 pointwise DeAR | — | — | 0.7411 | 0.0000 |
| B. 全局 KL | 0 | 关闭 | **0.7419** | **+0.0008** |
| C. 全局 + 局部 KL | 1.0 | 关闭 | **0.7419** | **+0.0008** |
| D. 全局 + 局部 KL + gate | 1.0 | 开启 | 0.7414 | +0.0003 |

B 首次计算并写入 1,290 条 RankLLaMA score；C/D 均为 `1290/1290` cache hit。B 中 2/43 个 query 改变 Top-10 顺序、1/43 个 query 改变 Top-10 候选集合；C 相同。D 的平均 teacher confidence 仅 `0.0391`，所以 gate 将平均最大实际残差压缩到约 `2.1e-6`，结果接近 raw；B/C 的平均最大实际残差分别约为 `3.5e-4`/`1.1e-3`。这一提升很小，且 DL19 已被用于选择 B/C，不能把 `0.7419` 当成无偏最终结果；应在独立开发集固定“是否启用局部 KL/gate”后再报告最终 DL19。

结果目录：

- A：`results/dear-p-3b-sll-kk-new-distribution-tta-raw-control/`
- B：`results/dear-p-3b-sll-kk-new-distribution-tta-global-k30-g0.10-s10/`
- C：`results/dear-p-3b-sll-kk-new-distribution-tta-global_local-k30-g0.10-s10/`
- D：`results/dear-p-3b-sll-kk-new-distribution-tta-global_local_gate-k30-g0.10-s10/`

每个 TTA 目录含 `evaluation_results.log`、`ranked_dl19.trec`、`dl19.txt.distribution_tta_diagnostics.jsonl` 与 `.summary.json`。可复跑：`MODE=global|global_local|global_local_gate bash run_distribution_tta_dl19.sh`；用 `TTA_TEACHER_CACHE_PATH` 指定复用的缓存。

## 基于留出熵一致性的双分支查询级 TTA（2026-09-27）

本轮的原始分支明确固定为 **DeAR + KK + SLL 已训练 checkpoint**：

```text
/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-new
```

这里的 `s0` 不是未训练的 Llama、不是仅 pointwise DeAR，也不是此前的 TS/TTA 分支；它是该 checkpoint 的原始 DeAR+KK+SLL pointwise logit 排序。KK/SLL 只在历史训练中形成 checkpoint，本轮没有重新训练、关闭或修改任何训练 loss。

### 实现

新增独立开关 `--use_entropy_hard_gate_tta`，与旧 pairwise TTA、分布 TTA、TS-SetRank、setwise residual 互斥。DeAR+KK+SLL 先对 Top-100 推理，提取 classification head 实际读取的最后有效 token hidden state；`modeling.py` 的 `region_ids==2` 与 classifier position 一致性检查仍然生效。DeAR 和 RankLLaMA 均冻结，`z/h/teacher score` 全部 detach，唯一有梯度的对象是每个 query 内重新零初始化的 FP32 向量 `u_q`。

对原始 Top-30 构造奇偶折：`A=(1,3,...,29)`、`V=(2,4,...,30)`。`u_A` 只以 A 上的 `KL(P_T||P_adapt)+anchor+u_norm` 优化，在 V 上计算 `Delta_H_A=H0(V)-H_A(V)`；`u_V` 独立从零优化，只在 A 上计算 `Delta_H_V`。主门控仅在两者均严格大于 0 时选择完整重优化后的 `s1`，否则直接选择 `s0`。没有任何 base/adapted 分数加权、teacher 分数融合或 qrels 参与推理。

输出分数使用原始 logit 尺度：`s0_raw=z`，`s1_raw=z+sigma_z*r`；它与公式中的 `zhat+r` 严格正仿射等价，排序不变，同时保证被拒绝 query 的输出与原始 DeAR+KK+SLL 分支完全相同。审查中修复了两项会伪造增益的问题：

1. 在小 KL 尺度上 Adam 首步可能使目标上升；现对临时 `u_q` 做回溯，必要时退回负梯度方向，且只保存不增大完整折内目标的迭代。
2. 不再将熵门控 run 统一改写为严格递减 TREC rank score；该做法会改变原始 score ties。现在直接输出完整 `s0` 或完整 `s1` 的真实分数。

RankLLaMA 缓存的 key 为 `(qid, docid, teacher checkpoint absolute path, candidate_k)`；旧的无 `candidate_k` cache record 会被安全跳过并增量重算。缓存命中不会依赖候选数组位置。

### 验证与 DL19

- GPU 检查：A800 80GB 空闲约 79GB；未发现训练进程；`/hy-tmp` 余量约 96GB。
- `py_compile`、`bash -n` 通过；CPU 单元测试验证奇偶划分、熵范围、`gamma=0`/`steps=0` 回退 base、独立 `u`、硬分支身份、少候选动态处理、qid/docid/checkpoint/candidate-K cache 对齐。
- 2-query smoke：首次 teacher cache 写入 60 条，复跑为 `60/60` 命中；两折优化 loss 下降，若留出熵不改善则正确拒绝适配；无 NaN/Inf。
- 全量 cache：首次 1,290 条 miss/写入；后续 B/C/D 均为 `1290/1290` hit。

| 组别 | 分支路由 | 选择 base/adapted | nDCG@10 |
|---|---|---:|---:|
| A | 原始 DeAR+KK+SLL `s0` | 43 / 0 | 0.7411 |
| B | 强制完整 `s1` | 0 / 43 | **0.7419** |
| C | 单折 A→V 熵硬选择 | 13 / 30 | **0.7419** |
| D | 双折 A→V 且 V→A 熵硬选择（主方法） | 17 / 26 | **0.7419** |

D 的平均 `Delta_H_A=-4.71e-6`、`Delta_H_V=-8.01e-6`；均值为负不等于每条 query 都为负，故仍有 26 条同时通过两折。D 中 Top-1 改变 0 条、Top-10 内部顺序改变 2 条、Top-10 集合改变 1 条，数值回退 0 条。

独立 qrels 后处理的 **Oracle upper bound / diagnostic only**：43 条中 `s1` 胜 1 条、`s0` 胜 0 条、平 42 条；`mean(max(nDCG(s0),nDCG(s1)))=0.7419`，与 D 实际值相同。qrels 仅在 `analyze_entropy_hard_gate_oracle.py` 中、所有 run 已生成后读取，推理模块未读取 qrels。说明在该配置下适配分支本身只有一条 query 的可获益空间，而双折熵门控恰好没有遗漏它；这不能证明“低熵必然意味着更高 nDCG”，也不能将已经在 DL19 上作出的设置选择当成无偏测试结论。

相关结果目录：

- A：`results/dear-p-3b-sll-kk-new-entropy-hard-raw-k30-g0.10-s10/`
- B：`results/dear-p-3b-sll-kk-new-entropy-hard-force-k30-g0.10-s10/`
- C：`results/dear-p-3b-sll-kk-new-entropy-hard-single-k30-g0.10-s10/`
- D：`results/dear-p-3b-sll-kk-new-entropy-hard-double-k30-g0.10-s10/`
- Oracle：`results/dear-p-3b-sll-kk-new-entropy-hard-oracle-analysis.json`

复跑：`MODE=raw|force|single|double bash run_entropy_hard_gate_tta_dl19.sh`。

## 结果文件位置

## Setwise contextual head 的连续多上下文残差推理（2026-09-26）

本轮使用完成首轮 SLL+KK 续训且 raw DL19 达到 `0.7435` 的 checkpoint：

```text
/hy-tmp/DeAR/outputs/dear-p-3b-sll-kk-setwise-sll-new
```

训练目标没有改动，仍为 `L_DeAR + 0.02 L_KK + 0.1 L_SLL`。修改的是一个**独立于原 TTA 和 TS-SetRank 的推理分支**：DeAR 对 100 篇候选 pointwise 打分并缓存 classifier-position 表示；仅取初始 Top-30，调用已训练的 setwise head。对每个 query 构造三轮、每轮 6 组、每组 5 篇的上下文：

1. 初始 DeAR rank 连续分组；
2. rank 循环移位 2 位后再分组，改变跨组邻接关系；
3. 初始 Top-15 与 Rank 16--30 交替排列后分组，强制头部与边界候选共同参与比较。

每轮中每篇 Top-30 文档恰好出现一次，三轮总共恰好获得 3 个 head logit。每组先按组内均值/总体标准差 z-score，再对同一文档三次结果平均为 `c_i`；仅在初始 Top-30 内以 `s_i=z_hat_i+gamma*c_i` 排序，Rank 31--100 保留原始 DeAR 顺序。最终按该受限顺序写入严格递减 TREC score，避免评测器按不同 score 尺度将范围外候选插入 Top-30。

### 实现审查

- 使用的向量是同一个 sequence-classification head 实际读取的最后有效 token 表示；运行时 `region_ids==2` 与 classifier position 的逐样本一致性检查仍在模型中执行。
- 该表示在 SLL 训练和新推理中均为未 L2-normalize 的 raw classifier hidden state；没有误用旧 TTA 的 normalized 表示。
- `setwise_head.bin` 在 DeAR GPU 评分结束、显存释放后仅在 CPU 加载一次，43 个 query 复用；没有再次运行 3B backbone，也没有让 head 每条 query 重载。
- trace 审查通过：DL19 共 43 个 query，每个 trace 有 18 个组，每篇 Top-30 文档 `observations=3`；Top-31--100 没有参与或被混回 setwise 排序范围。
- `python -m py_compile` 通过：`arguments.py`、`modeling.py`、`inference.py`、`setwise_residual.py`；30 文档合成单元测试确认三轮均无重复、无遗漏。

### DL19 结果

| 推理方式 | gamma | nDCG@10 |
|---|---:|---:|
| 该 checkpoint 的纯 pointwise DeAR | — | 0.7435 |
| 旧 Top-2 二值化 + TS-SetRank | — | 0.7318 |
| 连续多上下文残差 | +0.100 | 0.7377 |
| 连续多上下文残差 | +0.010 | 0.7422 |
| 连续多上下文残差 | -0.010 | 0.7435 |
| 连续多上下文残差 | -0.050 | 0.7437 |
| 连续多上下文残差 | -0.100 | 0.7433 |
| 连续多上下文残差（最佳扫描值，官方复核） | **-0.034** | **0.7443** |

因此新方案相对 raw pointwise 基线增加 `+0.0008`，并显著优于旧二值 TS 反馈；但**没有达到 0.747**。`gamma` 是在 DL19 上扫描得到的，不能把 `0.7443` 当作无偏最终报告；应在独立开发集固定 `gamma` 后再对 DL19 做一次最终评测。当前 checkpoint 的 setwise 分支有弱的局部信号，但不足以通过继续扩大残差获得 0.747；正向 `gamma` 明显伤害排序，而小幅负向 `gamma` 才有微弱收益，说明 SLL teacher-listwise 方向与 DL19/DeAR 的局部误差并未充分一致。

相关文件：`pointwise_reranker/setwise_residual.py`、`eval_setwise_continuous_residual_dl19.sh`；最佳结果目录为 `results/dear-p-3b-sll-kk-setwise-sll-new-contextres-k30-r3-gm0.034-dl19/`，其中包含完整 43-query 分组/分数 trace。

- 无 TTA 基线：`/tmp/dear-kk-new-raw-dl19.trec`，`nDCG@10=0.7350`。
- 强 TTA：`results/dear-p-3b-kk-new-tta-dl19/`，`nDCG@10=0.7143`。
- `gamma=0.05, steps=3, anchor=1.0`：`results/dear-p-3b-kk-new-tta-conservative-dl19/`，`nDCG@10=0.7353`。
- `gamma=0.10, steps=5, anchor=0.1`：`results/dear-p-3b-kk-new-tta-r4-g0.10-s5-lr1e-3-a0.1-dl19/`，`nDCG@10=0.7353`。
- `gamma=0.10, steps=5, anchor=0.3`：`results/dear-p-3b-kk-new-tta-r4-g0.10-s5-lr1e-3-a0.3-dl19/`，`nDCG@10=0.7353`。
- `gamma=0.20, steps=5, anchor=0.1`：`results/dear-p-3b-kk-new-tta-r4-g0.2-s5-lr1e-3-a0.1-dl19/`，`nDCG@10=0.7353`。
- teacher 一致性筛选：`results/dear-p-3b-kk-new-tta-r4-g0.10-s5-lr1e-3-a0.1-teachertrue-dl19/`，`nDCG@10=0.7353`。
- top-3 硬保护：`results/dear-p-3b-kk-new-tta-r4-g0.10-s5-lr1e-3-a0.1-teacherfalse-protect3-top10g-1-dl19/`，`nDCG@10=0.7353`。
- 平滑分段保护：`results/dear-p-3b-kk-new-tta-r4-g0.05-s5-lr1e-3-a0.1-teacherfalse-protect3-top10g0.01-dl19/`，`nDCG@10=0.7353`。
- 分类位置向量 `u_q`：`results/dear-p-3b-kk-new-tta-classifier-u-g0.10-s5-lr1e-3-a0.1-teacherfalse-protect0-top10g-1-dl19/`，`nDCG@10=0.7350`。
- 分类位置向量 + teacher Top-30 纠错：`results/dear-p-3b-kk-new-tta-classifier-u-g0.10-s5-lr1e-3-a0.1-teachertrue-corrtrue-k30-dt0.1-c1.0-protect0-top10g-1-dl19/`，`nDCG@10=0.7353`。
- SLL+KK checkpoint + 同一 TTT：`results/dear-p-3b-sll-kk-new-tta-classifier-u-g0.10-s5-lr1e-3-a0.1-teachertrue-corrtrue-k30-dt0.1-c1.0-dl19/`，`nDCG@10=0.7419`。
- TS-SetRank Mock 无反馈控制：`results/dear-p-3b-sll-kk-new-ts-setrank-mock-control-k100-b5-bs10-e2-dl19/`，`nDCG@10=0.7414`；trace：`dl19.txt.ts_setrank_trace.jsonl`。
- TS-SetRank + Llama-3.2-3B 基座 judge：`results/dear-p-3b-sll-kk-new-ts-setrank-llama-base-id-k100-b5-bs10-e2-dl19/`，`nDCG@10=0.7376`；trace：`dl19.txt.ts_setrank_trace.jsonl`。
