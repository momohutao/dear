# MS MARCO intent hard negative mining

> 本文记录旧的离线规则筛选流程。当前训练使用原始完整池；新的冻结学生 top1、教师过滤和运行时注入见 [STUDENT_TEACHER_MINING.md](STUDENT_TEACHER_MINING.md)。

```bash
python mine_intent_hard_negatives.py \
  --input /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --output /hy-tmp/DeAR/data/msmarco-passage/train_intent_hard.jsonl.gz \
  --seed 42 --hard-slots 4
```

输出保留输入 JSONL schema、query 顺序和 passage 内容，每条严格为
**1 positive + 7 negative**。候选充足时 7 个 negative 不重复；候选不足时沿用
现有 trainer 的有放回采样，统计 `short_pool_rows` 和 `repeated_output_slots`。
正例从原有正例中按 query seed 固定选一个；
原有全部 positive docid 和正文副本都从 negative 池排除。
这会固定正例，失去原训练集多正例轮换的多样性，是要求 1+7 的取舍。

只依赖 Python 标准库，CPU 流式处理，不使用 GPU，也不读取 DL19 qrels。
原数据中的每 query negative 池是检索范围：默认约 30 个候选筛成 7 个。
这是候选池内 mining，不是对完整 MS MARCO passage collection 重新检索。
池中不存在的 hard negative 无法通过本脚本发现。

## 三类可审计规则

每个候选首先需要与 query 共享主题词（至少 min(2, query 内容词数) 个，覆盖率 ≥ 0.35）。

- `keyword_constraint_missing`：query 的方向、年龄、性别、时间顺序、数字、
  政策名称等限定词出现在选中的 positive 中，但 negative 缺少该约束。
  常见年龄/性别别名会归一处理。词表是有限启发式，不是通用实体识别。
- `domain_confusion`：从 query 推断领域；query 不明确时从 positive 推断，
  negative 中另一个领域至少命中 2 个词且比目标领域多 2 个词。
  当前覆盖医学、建筑、计算机、生物、金融、烹饪和法律；医学/生物学视为兼容。
  其他混合领域时仍可能误判。
- `intent_mismatch`：识别列表、比较、原因、步骤、时长、数量、地点和定义意图。
  先要求 positive 含目标回答形式标记；negative 缺少该标记，
  却出现其他回答形式的标记时才触发。
  此规则只识别表面语言模式，不能证明一个段落没有回答 query。

同一个 candidate 可以有多个标签。选择时按 domain / intent / keyword 轮询，
每类内部按 query 词覆盖率 + 0.1 × 标签数排序，docid 稳定打破并列。
默认最多选择 4 个此类候选，然后从固定 seed 的原候选随机 7 个基线中补齐。
实际 hard 标签数量可能超过 4，因为随机补齐的候选也可能带有 hard 标签。
`--hard-slots 0` 可产生完全相同口径的随机基线。

所有规则标签是**待核验的 mining 假设**，不是 teacher score 或真实相关性判断。
只使用原有 negative 标签且排除已知 positive；仍可能有未标注相关文档。
不能把文本匹配率高直接解释为真正的 hard negative，更不能保证提升 DL19。

## 统计与审计

输出三个文件：

- `train_intent_hard.jsonl.gz`：训练输入。
- `train_intent_hard.jsonl.gz.stats.json`：pool（去重、排除正例后的全部候选）、
  before（每 query 随机 7 个）、after（选出的 7 个）三种分布，替换数、过滤数。
- `train_intent_hard.jsonl.gz.audit.jsonl`：前 100 个 query 的候选标签、原因、
  替换前后 docid；`--audit-rows` 可改变行数。

主类型优先级为 domain > intent > keyword > other，每个 doc 只计一次，比例和为 1。
另提供多标签计数，重叠标签不能相加解释为 doc 数。
before/after 分母相同，都是输出 query 数 × 7；before 是脚本的确定性随机基线，
不是对历史训练中实际抽到的 batch 的复原。
不同 seed/query 下均独立采样，脚本复跑 gzip 数据文件字节一致；统计包含运行耗时。

候选不足 7 个时保持随机有放回基线，不专门重复最难的候选。
`--short-pool error` 可要求每条必须有至少 7 个不同候选。
没有可用 negative、无正例或格式错误时默认报错，不产生最终训练文件。
`--invalid skip` 可以显式跳过，并在统计/审计记录；此时输出 query 数会减少。
不把别的 query 的未判断文档自动当负例。已有输出文件不覆盖。

先抽样检查：

```bash
python mine_intent_hard_negatives.py --input /hy-tmp/DeAR/data/msmarco-passage/train.jsonl.gz \
  --output /tmp/train_intent_hard_sample.jsonl.gz --limit 1000
python -m unittest test_intent_hard_mining.py
```

训练时指向新 JSONL，继续设置 `--train_n_passages 8`。现有 trainer 已按 1+7 分组，
无须修改 loss。脚本不会自动切换训练数据或启动训练。
