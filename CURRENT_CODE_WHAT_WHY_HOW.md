# 当前代码思路：DeAR 区域关系蒸馏

## 一句话概括

当前方案用冻结的 Llama2-13B RankLLaMA 教师指导 Llama-3.2-3B 学生：除了学习最终排序分数，还让学生模仿教师在 Query、Document、Score Position 三个语义区域之间的内部关系；训练参数通过 DL19、DL20 上的三种子结果进行顺序贪心选择。

---


# What：现在的代码在做什么

## 1. 任务与模型

这是一个 passage reranker。每条训练样本包含一个 Query、一个正例 passage 和一个负例 passage，学生模型分别对 Query–Passage 对打分，然后学习让正例排在负例前面。

- 学生模型：Llama-3.2-3B
- 教师模型：Llama2-13B RankLLaMA teacher
- 学生训练方式：LoRA
- 教师状态：完全冻结，只提供分数和内部关系监督
- 训练数据：MS MARCO Passage
- 测试集：TREC DL19、DL20、beir
- 主评测指标：nDCG@10

学生使用 LoRA 更新以下模块：

```text
q_proj, v_proj, o_proj,
down_proj, up_proj, gate_proj
```

LoRA 配置为：

```text
r = 32
lora_alpha = 64
lora_dropout = 0.1
```

## 2. 输入区域

代码不依赖学生和教师使用相同 tokenizer，而是在两侧分别把输入标记为三个语义区域：

```text
Q = Query 区域
D = Document 区域
S = Score Position
```

模型输入逻辑上为：

```text
[Query tokens] [Document tokens] ... [Score Position]
```

Score Position 固定在 padding 后序列的最后一位。序列分类头读取这一位置的隐藏状态并输出相关性分数。这样既保持原 DeAR 的打分方式，也让关系蒸馏拥有一个明确的“最终决策位置”。

## 3. 总损失

当前总损失为：

```text
L_total = L_DeAR
        + lambda_margin_mse * L_margin
        + lambda_rat        * L_RAT
        + lambda_down       * L_down
```



### 3.1 DeAR 主损失(baseline)

```text
L_DeAR = (1 - alpha) * L_CE + alpha * L_KD
```

- `L_CE`：以第 0 个 passage 为正例的组内 softmax cross entropy。
- `L_KD`：学生与教师组内分数分布之间的 KL 蒸馏损失。
- 蒸馏温度 `T=2`，KL 项乘以 `T²`。
- 当前 `alpha=0.1`，即监督排序损失占主要权重，教师 logit 蒸馏作为辅助。

### 3.2 Margin 损失

分别计算学生和教师的正负例分差：

```text
student_margin = student_positive - student_negative
teacher_margin = teacher_positive - teacher_negative
```

然后使用 Smooth L1：

```text
L_margin = SmoothL1(student_margin, teacher_margin)
```

它约束的不只是“谁排在前面”，还约束正负例之间应拉开多大距离。

### 3.3 RAT：Q–K 区域关系蒸馏

代码通过 forward hook 捕获指定层的 `q_proj` 和 `k_proj` 输出，并完成以下处理：

1. 按模型原始 attention heads 拆分 Q/K。
2. 对 Q/K 显式应用 RoPE。
3. 按真实 GQA 规则把 K heads 展开到与 Q heads 对齐。
4. 将不同规模模型的原始 heads 重构成统一的 8 个 relation heads。
5. 计算带 causal mask 的 token-to-token Q–K 关系分布。
6. 将 token 关系聚合为 Q、D、S 三个区域间的 `3 × 3` 关系矩阵。
7. 用 `KL(student || teacher)` 约束学生关系分布。

区域源权重为：

```text
Query source    = 0.00
Document source = 0.25
Score source    = 1.00
```

因此训练重点放在“文档如何汇聚信息”以及“最终打分位置如何读取 Query/Document 信息”，尤其强调 Score Position 的决策关系。



### 3.5 down-proj 区域关系损失

代码还会捕获最后层 MLP 的 `down_proj` 输出：

1. 分别对 Q、D、S 区域内的 token 表示做平均池化。
2. 在学生和教师各自的隐藏维度中构造区域相似度矩阵：

```text
R(H) = softmax(H H^T / sqrt(d))
```

3. 对师生的 `3 × 3` 区域关系矩阵计算 MSE。

这里比较的是关系矩阵，而不是直接比较隐藏向量，所以即使学生和教师隐藏维度不同，也可以进行对齐。

## 4. QKNorm 的当前状态

代码实现了最后一层 QKNorm：对 Q/K 每个 head 做 L2 normalization，并为每个 Query head 学习一个正尺度。

但当前调参命令为：

```text
last_layer_qknorm = false
```

所以当前结果不包含 QKNorm 的作用。它是可选实验分支，不应算入正在运行方案的有效组件。

---

# Why：为什么这样设计

## 1. 只蒸馏最终分数，监督信息太少

传统 logit KD 只告诉学生教师最终给了多少分，却没有告诉学生教师如何在 Query 和 Document 之间组织信息。对于 3B 学生模仿 13B 教师，单一标量分数容易成为过弱监督。

因此代码同时蒸馏：

- 最终排序分布；
- 正负例 margin；
- Q/K attention 关系；
- MLP down-proj 的区域语义关系。

这些信号分别覆盖输出层、排序间隔、attention 路径和 MLP 表示结构。

## 2. 使用区域关系，解决师生结构不一致

学生和教师存在多种不一致：

- 参数规模不同；
- hidden size 不同；
- attention head 数不同；
- GQA 结构可能不同；
- tokenizer 不同，token 无法逐位置严格对应。

直接做 token hidden-state MSE 容易产生错误对齐。当前方案先在各自 tokenizer 空间中计算 token 关系，再聚合到共享的 Q/D/S 语义区域。最终两边都得到固定大小的区域关系矩阵，因此可以跨模型、跨 tokenizer 对齐。

## 3. 强调 Score Position，贴近最终排序决策

RAT 的源区域权重将 Score Position 设为 1.0，将 Document 设为 0.25，而 Query 设为 0。核心假设是：reranker 最重要的内部行为，是最终打分位置如何从 Query 和 Document 汇聚证据。

这比平均约束所有 token 关系更贴近排序任务，也能减少与最终决策关系较弱的约束噪声。

## 4. 使用 LoRA，降低实验成本

学生只训练 LoRA 和分类头，教师完全冻结。这样可以：

- 降低显存需求；
- 缩短每个候选参数的训练时间；
- 方便保存和切换 checkpoint；
