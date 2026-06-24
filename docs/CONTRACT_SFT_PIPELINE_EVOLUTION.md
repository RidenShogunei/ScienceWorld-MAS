# Contract 版本 SFT 数据生成管线演进

本文梳理 ScienceWorld-MAS 中 Contract 版本数据管线从最初设想到当前
`semantic expert rollout` 版本的演进过程。重点不是罗列脚本，而是说明：

1. 每一版数据到底来自哪里。
2. Main 和 Sub 分别学习什么。
3. 为什么上一版不够用。
4. 当前版本解决了什么，仍然没有证明什么。

## 1. 我们最终要训练的接口

系统采用固定的分层 Main/Sub 架构：

```text
Task + Environment State
          |
          v
Main planner
          |
          | structured contract
          v
Sub executor
          |
          | executable ScienceWorld action
          v
ScienceWorld environment
          |
          +---- observation / reward / score ----> Main and Sub
```

Main 不直接执行环境动作，而是输出一个短期、可检查的执行契约：

```text
[contract]{
  "subgoal": "Move to the kitchen",
  "success_condition": "The agent is in the kitchen or has opened the path to reach it.",
  "target_objects": ["door", "hallway", "kitchen"],
  "action_guidance": [
    "open door to hallway",
    "go to hallway",
    "open door to kitchen",
    "go to kitchen"
  ],
  "handoff_if": "complete when success_condition is met; need_replan if blocked"
}[/contract]
```

Sub 接收 Contract、当前 Observation、最近执行历史和环境给出的
`Valid actions`，每次只输出一个真实动作：

```text
[action]go to kitchen[/action]
[subtask_done]true[/subtask_done]
[handoff]complete[/handoff]
```

这里的核心训练目标有两个：

- Main 学会根据当前状态生成有语义、可执行、可验收的短期 Contract。
- Sub 学会在 Contract 约束下，从当前合法动作中选择下一步 action，并在合适
  的时机继续、完成、阻塞或请求重新规划。

## 2. 三个容易混淆的维度

判断一份数据是否“真实”或“可用”时，需要把下面三个维度分开。

| 维度 | 可能取值 | 含义 |
| --- | --- | --- |
| 状态来源 | 静态数据 / 环境 rollout | Observation 是否来自真实逐步执行后的环境状态 |
| 动作来源 | 专家 gold / 模型生成 | 实际 action 是官方正确轨迹，还是模型现场探索 |
| Contract 来源 | LLM 蒸馏 / 规则包装 / 模型在线规划 | Main 标签如何得到 |

因此，“从环境跑出来”并不等于“Main 自己规划成功”；“动作 100% 合法”也不
等于“Contract 具有可泛化的规划语义”。

## 3. 演进总览

| 阶段 | 时间 / 提交 | 状态来源 | 动作来源 | Contract 来源 | 主要结果 |
| --- | --- | --- | --- | --- | --- |
| V0 原始分层 SFT | Contract 之前 | Multi-Square 静态轨迹 | 专家 | 单行 subtask | 建立 Main/Sub 基线，但通信信息不足 |
| V1 Verbose Contract 蒸馏 | 2026-06-16 `83c0282` | Multi-Square 静态轨迹 | 专家 | Kimi 离线蒸馏 | Contract 成型，但字段过多且训练/部署上下文不一致 |
| V1.1 审计与混合 | `09d7148`, `92e07ea` | 静态 + 少量在线 | 专家 + 模型 | 蒸馏 + 在线 | 加入格式审计和 native 数据，但数据分布仍混杂 |
| V2 Native 模型 rollout | `8360b25` 起 | 真实环境 | Kimi | Kimi 在线规划 | 接口最真实，但成功率低、循环多、成本高 |
| V2.1 Action-ID 弯路 | `2be6d9e` | 真实环境 | Kimi 选 ID | Kimi 在线规划 | 提高局部可执行率，但训练接口偏离最终 action 接口 |
| V3 Minimal Contract | 2026-06-20 `d71b374` | Multi-Square 静态轨迹 | 专家 | Kimi/规则蒸馏 | Contract 更清爽、格式稳定，但 Sub prompt 仍缺环境上下文 |
| V4 Minimal Native MiniMax | `6f0262f` 至 `0194e14` | 真实环境 | MiniMax | MiniMax 在线规划 | Prompt 对齐，但 50 集成功率仅 6% |
| V5 Expert Environment Replay | `44e504e`, `245286a` | 真实环境 | ScienceWorld gold | 动作序列包装 | 50/50 成功，Sub 完全对齐；Main 标签退化为专家动作复述 |
| V6 Semantic Expert Rollout | 2026-06-24 `0cb48ec` | 真实环境 | ScienceWorld gold | 语义规则包装 | 当前候选版：30 task 全覆盖、30/30 成功、接口完全对齐 |
| V7 Expert-Subtask Contract | 2026-06-24 | Multi-Square 专家状态 | Multi-Square gold | 原始 expert subtask 的无损增强 | 当前推荐版：保留因果规划标签，剔除明确上游冲突 |

## 4. V0：单行 Subtask 的原始分层 SFT

最初的数据直接使用 Multi-Square：

```text
Main: task + planner state -> [subtask]...[/subtask]
Sub:  subtask + observation -> [action]...[/action]
```

这版证明了 Main/Sub 分开训练和环境评测链路可以工作，但单行 subtask 缺少：

- 成功条件；
- 目标对象；
- 可接受的动作方向；
- 何时交还 Main；
- 阻塞时如何处理。

Contract 管线由此产生：不是改变 Main 负责拆分、Sub 负责执行的架构，而是把
二者之间的简单自然语言指令升级为有明确接口的通信协议。

## 5. V1：Verbose Contract 离线蒸馏

脚本：`generate_contract_sft_data.py`

该版本读取 Multi-Square 专家轨迹，让 Kimi 根据 task、observation、原
subtask 和专家动作生成九字段 Contract：

```text
goal, subgoal, rationale, target_objects, location_hint,
required_tools, success_condition, action_guidance,
fallback_if_blocked
```

设计上有一个正确且一直保留的约束：

> LLM 只负责蒸馏通信语义，不得改写低层动作标签；Sub action 始终来自专家轨迹。

随后加入 `audit_contract_sft.py`，检查 Contract 可解析率、字段完整率，以及
`action_guidance` 是否与专家动作对齐。

### 这一版解决了什么

- Main/Sub 之间第一次有了结构化 Contract。
- 专家动作仍可审计，不会被蒸馏模型“自由发挥”污染。
- 数据可以批量生成、缓存、失败重试和审计。

### 为什么没有停在这里

- 九字段过重，包含 rationale 等不直接服务执行的内容。
- 数据仍是静态 teacher forcing，不是部署时的逐步环境状态。
- Sub 训练 prompt 与环境评测 prompt 不一致。
- 离线 exact match 高，不代表动作在真实状态中可执行。

## 6. V1.1：离线蒸馏与 Native 数据混合

`data/sft_contract_native_v1` 曾将两类数据混合：

- `contract_v2`：3253 条，专家动作 + Kimi 蒸馏 Contract；
- `native_handoff_v2`：950 条，Kimi 环境 rollout 中筛出的局部有效步骤。

总计 4203 条，其中 Main 936 条、Sub 3267 条。

这一步认识到了环境数据的重要性，但混合本身没有消除分布差异：

- 静态数据中的 Sub 看见的是 gold contract 和专家上下文；
- native 数据中的 Sub 看见的是模型 contract 和偏离专家轨迹后的状态；
- 局部非负过滤会改变 benchmark 的自然状态分布；
- Main 和 Sub 的数据质量不能用同一条过滤规则保证。

因此，“格式相同可以拼接”不等于“语义分布相同可以直接训练”。

## 7. V2：模型原生环境 Rollout

脚本：

- `collect_kimi_mas_rollouts.py`
- `generate_sft_from_kimi_rollouts.py`

流程变成：

```text
模型 Main 现场生成 Contract
        |
模型 Sub 读取真实 Observation 和 Valid actions
        |
模型 Sub 输出 action
        |
ScienceWorld 执行动作并返回下一状态
        |
Rollout 转为 Main/Sub SFT 样本
```

这是第一条真正与部署流程一致的生成管线。Sub prompt 开始包含：

- Contract；
- 当前 Observation；
- Recent execution history；
- 完整 Valid actions。

同时增加 `max_steps_per_subtask` 和 `handoff`，防止一个失效 Contract 吃掉整集
步数，并允许 Sub 主动请求 Main 重新规划。

### Action-ID grounding 为什么被放弃

中间版本曾把合法动作编号为 `A0`, `A1`, ...，让 Sub 输出：

```text
[action_id]A3[/action_id]
```

它确实可以降低“意思对但字符串不可执行”的情况，但也引入了新的问题：

- ID 只在当前候选列表中有意义；
- 训练目标不再是最终部署需要的真实 action；
- 候选排序或截断会改变 ID 的含义；
- 模型容易学成列表索引器，而不是环境动作执行器。

最终原则改为：

> 如果环境已经提供合法动作，就让 Sub 直接复制真实 action；不要再训练一个
> 部署时并不需要的 action-id 中间接口。

当前默认也不再过滤、排序或截断 benchmark 提供的合法动作。相关开关只保留
给调试和消融实验。

### 为什么 Native Rollout 不能独立承担专家数据生产

真实 rollout 的接口最准确，但生成模型会循环、误规划和提前耗尽步数。使用
MiniMax 跑出的 50 集 minimal native 数据结果为：

| 指标 | 数值 |
| --- | ---: |
| 成功率 | 6% |
| 平均分 | 1.34 |
| Main 格式合法率 | 99.28% |
| Sub 格式合法率 | 100% |
| Action valid | 94.62% |

这说明问题已经不主要是格式，而是任务规划和长程执行能力。继续放大同类 rollout
会产生大量“接口真实但策略失败”的样本，不能作为高质量正向 SFT 的主体。

## 8. V3：Minimal Contract

脚本：`generate_minimal_contract_sft_data.py`

Verbose Contract 被压缩为五个真正影响执行的字段：

```text
subgoal
success_condition
target_objects
action_guidance
handoff_if
```

其中：

- Kimi/MiniMax 只蒸馏 `subgoal`、`success_condition`、`target_objects`；
- `action_guidance` 由程序从专家动作中提取；
- `handoff_if` 使用固定规范；
- Sub action 仍由专家标签提供。

`minimal_contract_sft_v1_sample1000` 包含：

| 类别 | 数量 |
| --- | ---: |
| Main | 1000 |
| Sub | 4691 |
| 总计 | 5691 |

审计结果全部为 100%，说明 schema 和标签构造是稳定的。

### 这一版暴露出的关键问题

训练时 Sub 的输入主要是：

```text
Contract + Observation
```

环境评测时却是：

```text
Contract + Observation + Valid actions + Recent history
```

因此 Sub 离线指标很好，进入环境后却可能第一步就解析或选动作失败。这不是模型
“没训练 Sub”，而是数据生成 prompt 与部署 prompt 不同。

## 9. V4：Minimal Native MiniMax Rollout

为了消除上述 prompt mismatch，native collector 支持 minimal schema，并切换
到 MiniMax API。生成的
`data/kimi_mas_sft_minimax_minimal_native_train50` 有：

| 类别 | 数量 |
| --- | ---: |
| Main | 397 |
| Sub | 1059 |
| 总计 | 1456 |

所有 Sub 样本都带 Valid actions 和 Recent history，格式层面已经对齐部署。

但由于动作和 Contract 都由模型在线生成，50 集只有 3 集成功。它适合用于：

- 收集真实失败状态；
- 做恢复、重规划和负轨迹研究；
- 作为少量 native 分布补充。

它不适合直接作为大规模正向专家 SFT 的主要来源。

## 10. V5：ScienceWorld Gold 环境回放

脚本：`collect_expert_minimal_rollouts.py`

我们随后把“真实环境状态”和“高成功率动作”结合起来：

1. ScienceWorld 以 `generateGoldPath=True` 提供官方 gold actions。
2. 每个 gold action 都在真实环境中逐步执行。
3. 每一步保存当时真实的 Observation、Valid actions、history、score 和 done。
4. 再把完整 rollout 转成与 native rollout 相同的 Main/Sub SFT 格式。

这不是静态复制数据。Observation 和 valid action 集合来自 gold action 执行后的
真实状态转移；只是动作选择不再交给能力不足的生成模型，而使用环境专家策略。

首批 50 集结果：

| 指标 | 数值 |
| --- | ---: |
| 成功率 | 100% |
| 平均分 | 100 |
| 平均步数 | 59.4 |
| Main/Sub 格式合法率 | 100% |
| Action valid | 100% |

### V5 仍然失败在哪里

最初 Contract 只是把每四个 gold action 包装为：

```text
Execute the next expert step sequence: action 1; action 2; action 3; action 4
```

这会让 Main 学到“看到专家未来动作后复述动作序列”，而不是：

```text
当前任务 + 当前状态 -> 下一阶段语义目标
```

所以 Sub 可以在 gold contract 下离线达到很高 exact match，Main 却无法在新状态
自主规划。环境评测中的主要瓶颈由此转移到 Main。

## 11. V6：Semantic Expert Rollout

当前版本仍使用真实环境 gold replay，但不再把动作序列直接写进 subgoal。
`semantic_goal()` 根据一个短 action chunk 推导语义目标和可检查成功条件，例如：

| Gold action 类型 | 生成的语义 subgoal |
| --- | --- |
| `open door`, `go to` | `Move to the kitchen` |
| `pick up` | `Collect thermometer, metal pot` |
| `examine`, `look at` | `Inspect chocolate` |
| `move`, `pour`, `mix` | `Place or combine ...` |
| `activate`, `open`, `close` | `Operate ...` |
| `wait` | `Wait for the current process to advance` |

`action_guidance` 仍保留该短期阶段内的专家动作，Sub 目标仍是当前状态下真实可执行
的 gold action。

当前首批数据：

- Rollout：
  `data/expert_semantic_minimal_rollouts/train_k1_success.jsonl`
- SFT：
  `data/expert_semantic_minimal_sft_train_k1_success/train.jsonl`
- 审计：
  `data/expert_semantic_minimal_sft_train_k1_success/audit_report.json`

结果：

| 指标 | 数值 |
| --- | ---: |
| Task type 覆盖 | 30/30 |
| Episode | 30 |
| 成功率 | 100% |
| 平均分 | 100 |
| 平均步数 | 49.03 |
| Main 样本 | 374 |
| Sub 样本 | 1471 |
| 总样本 | 1845 |
| Main/Sub parse rate | 100% |
| Sub action 位于 Valid actions | 100% |
| 旧动作序列式 subgoal | 0 |

## 12. V7：Expert-Subtask Contract

V6 环境评测暴露了一个关键问题：Gold action 是正确的执行监督，但从未来
Gold action 反推的 Contract 只是后验摘要，不一定能由 Main 当前可见的信息预测。

V7 回到 Multi-Square 原始 High-level/Low-level 对齐关系：

```text
Task + current planner state -> original expert subtask
                                      |
                                      v
                           minimal Contract enrichment

Contract + low-level observation + history -> original expert action
```

约束如下：

- `subgoal` 逐字保留原始 expert subtask，模型无权改写。
- MiniMax 只能根据 task、当前 planner state 和 expert subtask 补充
  `success_condition` 与 `target_objects`。
- enrichment 不读取未来 executor observation 或 gold action。
- `action_guidance` 仅根据 expert subtask 生成动作类型提示。
- Sub action、done 和 handoff 仍来自 Multi-Square Low-level gold 标签。
- 同一 High-level source trajectory 不跨 train/val/test。
- 明确的上游目标冲突会被剔除，例如 subtask 要求 green box、gold action 却移动
  到 blue box。

首批因果 Contract 数据位于：

```text
data/expert_subtask_contract_sft_v3_simple_minimax_sample1000/
```

包含 Main 1000 条、Sub 4696 条，共 5696 条；所有关键结构和 gold 标签一致性
审计均为 100%，source trajectory split leakage 为 0。

为保持与原始 baseline 的单变量对照，Sub 使用最简单的
`Contract + Observation -> action + done + handoff` 接口，不加入 Recent history
或 Valid actions。环境评测必须使用同一接口。

## 13. 当前推荐的数据生成管线

当前应把 V7 作为 Main 因果规划监督和静态专家 Sub 监督的主线。V6 环境 replay
可保留用于补充带完整 Valid actions 的 Sub 数据，但不再用于构造 Main 标签：

```text
读取并对齐 Multi-Square High/Low-level expert trajectories
                  |
                  v
保留原始 expert subtask 与阶段边界
                  |
                  v
仅用当前可见信息补充 minimal Contract
                  |
                  v
使用 Low-level gold action / done 构造 Sub 标签
                  |
                  v
补充可用的 Recent history
                  |
                  v
按 High-level source trajectory 划分 train/val/test
                  |
                  v
审计 subgoal/action 一致性与上游目标冲突
                  |
                  v
训练 Main/Sub 并做 stratified-145 环境评测
```

生成命令：

```powershell
python generate_expert_subtask_contract_sft.py `
  --provider minimax `
  --sample-size 1000 `
  --workers 6 `
  --output-dir data/expert_subtask_contract_sft_v3_simple_minimax_sample1000

python audit_expert_subtask_contract_sft.py `
  --input-dir data/expert_subtask_contract_sft_v3_simple_minimax_sample1000 `
  --output data/expert_subtask_contract_sft_v3_simple_minimax_sample1000/audit_report.json
```

## 14. 当前版本还没有证明什么

V7 是目前结构最合理的数据，但只能称为“候选可用”，还不能仅凭数据审计宣布
训练问题已经解决。

### 14.1 Contract 扩展字段仍是模型生成

`subgoal` 已经是因果专家标签，但 `success_condition` 和 `target_objects` 仍由
MiniMax 补充，可能存在措辞过度具体或世界知识推断偏差。

### 14.2 Valid actions 尚未作为独立消融验证

当前版本刻意复用原始 baseline 的自由 action generation，不使用 Valid actions。
未来可以在保持其他变量不变时单独加入候选列表，判断它是帮助 grounding，还是
因长列表稀释任务信息。

### 14.3 数据量仍是试验规模

当前只有 1000 条 Main 样本，应先做环境验证，再决定是否扩到全部 13700 余条
expert decisions。

### 14.4 Gold 标签一致不等于闭环成功

真正部署时 Contract 和 action 都由训练后的模型生成，仍必须通过闭环环境评测
和 Oracle Main/Sub 消融判断。

## 15. 扩量前的验收门槛

不要只看 val loss、parse rate 或 teacher-forcing exact match。下一轮应至少完成：

| 检查 | 目的 |
| --- | --- |
| Main 离线语义评测 | 检查是否生成与当前状态匹配的短期目标，而非模板串台 |
| Sub action-in-valid-actions | 检查输出是否属于当前环境合法动作 |
| Main-only oracle-Sub 消融 | 判断 Main 是否仍是主要瓶颈 |
| Oracle-Main + trained-Sub 消融 | 独立判断 Sub 的真实执行能力 |
| Stratified-145 闭环评测 | 覆盖全部 task type，避免只在已见任务上看起来有效 |
| In-domain / OOD 分组 | 区分记忆、覆盖不足和真正泛化 |

建议决策顺序：

1. 先用当前 k=1 数据训练一版 Main/Sub。
2. 跑统一的 offline、oracle 消融和 stratified-145。
3. 若 Main 仍弱，先改语义分段和 Contract 标注，不立即扩量。
4. 若闭环指标明确优于旧数据，再扩到 `k-per-task=3` 或 `5`。
5. Native 模型 rollout 保留为失败恢复和分布补充，不作为正向专家主体。

## 16. 当前形成的设计原则

经过这些版本，数据管线已经形成以下稳定原则：

1. Main 负责拆分任务，Sub 负责执行 action；Contract 只升级通信，不改变分工。
2. 训练 prompt 必须与部署 prompt 一致，尤其是 Valid actions 和 Recent history。
3. Sub 默认输出真实 action，不使用 action-id 作为最终训练接口。
4. Benchmark 提供的合法动作默认完整保留，不做主观过滤、排序或截断。
5. 正向专家数据优先使用真实环境状态 + gold action。
6. Contract 必须表达语义目标和成功条件，不能只是未来动作序列的改写。
7. 格式正确、动作合法、轨迹成功和闭环泛化必须分别评测。
8. 数据扩量之前先验证生成范式，避免把结构性错误批量复制。

## 17. 相关实现与产物

| 文件 | 作用 |
| --- | --- |
| `contract_schema.py` | Verbose Contract schema |
| `generate_contract_sft_data.py` | V1 离线 Contract 蒸馏 |
| `audit_contract_sft.py` | Verbose Contract 审计 |
| `generate_minimal_contract_sft_data.py` | V3 Minimal Contract 离线生成 |
| `audit_minimal_contract_sft.py` | Minimal Contract 审计 |
| `collect_kimi_mas_rollouts.py` | V2/V4 模型原生环境 rollout |
| `collect_expert_minimal_rollouts.py` | V5/V6 专家环境 replay 与语义 Contract |
| `generate_sft_from_kimi_rollouts.py` | 统一 rollout 到 Main/Sub SFT 的转换 |
| `generate_expert_subtask_contract_sft.py` | V7 因果 Expert-Subtask Contract 数据 |
| `audit_expert_subtask_contract_sft.py` | V7 标签一致性与 split 审计 |
| `rollout_schema.py` | MainDecision、SubInvocation、ActionStep 和 SystemRollout |
| `docs/CONTRACT_DISTILLATION.md` | 早期离线蒸馏说明 |
| `docs/NATIVE_KIMI_ROLLOUTS.md` | 模型原生 rollout 说明 |

## 18. 一句话结论

当前管线已经认识到 Gold execution 不等于 Gold planning。V7 不再从未来动作
反推 Main 标签，而是保留原始 High-level expert subtask，仅将其无损扩展成
Contract；下一步要验证的是，这种因果规划监督能否恢复原始 baseline 的 Main
优势，同时保留 Contract 通信接口。
