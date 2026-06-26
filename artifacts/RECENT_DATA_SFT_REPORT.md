# Contract 时代实验总结

> 基座 `Qwen/Qwen3.5-9B` · bf16 LoRA · 环境评测集 Stratified-145（145 ep，30 task）  
> 对照基线（contract 之前 subtask + RL）：mean **18.12**，success **9.0%**

---

## 一、每一版数据是什么样的

协议沿两条线演进：**Full Contract**（完整 JSON，含 goal / rationale / location_hint 等）→ **Minimal Contract**（精简 JSON：subgoal / success_condition / target_objects / action_guidance / handoff_if）。Sub 统一输出 `[action]...[/action][subtask_done]...[/subtask_done][handoff]...[/handoff]`。

---

### v1 · Full Contract 原生混合（`sft_contract_native_v1`）

| 项 | 内容 |
|----|------|
| **路径** | `data/sft_contract_native_v1/` |
| **规模** | train 3782 / val 421（Main 842 + Sub 2940） |
| **来源** | **contract_v2**（2927）：Kimi 蒸馏 full contract + 专家 action；**native_handoff_v2**（855）：Kimi MAS 环境 rollout |
| **Main 标签** | `[contract]{"goal","action_guidance","fallback_if_blocked","location_hint","rationale",...}[/contract]` |
| **Sub 标签** | 专家 action + handoff |
| **特点** | 两路 Sub prompt 分布差异大；Sub 样本 valid actions 列表常很长；训练 max_length 1024 |

---

### v1-clean · Main 标签清洗（`sft_contract_native_v1_mainclean`）

| 项 | 内容 |
|----|------|
| **路径** | `data/sft_contract_native_v1_mainclean/` |
| **相对 v1** | Sub 数据完全相同；仅 Main 标签被改 |
| **清洗规则** | `action_guidance` 去重、上限 8 条；collapse 重复 monitoring loop |
| **改动量** | train Main 842 条中 **199** 条被修改 |

---

### v2 · Minimal Contract 蒸馏（`minimal_contract_sft_v1_sample1000`）

| 项 | 内容 |
|----|------|
| **路径** | `data/minimal_contract_sft_v1_sample1000/` |
| **Schema** | `minimal_contract_v1` |
| **规模** | train 4777 / val 363 / test 551（Main 836 + Sub 3941） |
| **来源** | Kimi HTTP expert 500 步 rollout + 语义字段抽取 |
| **Main 标签** | `[contract]{"subgoal","success_condition","target_objects","action_guidance","handoff_if"}[/contract]` |
| **特点** | 字段更少、与后续环境 `--protocol minimal` 对齐；guidance 上限 6 |

---

### v3 · Kimi/MiniMax Native Rollout（`kimi_mas_sft_minimax_minimal_native_train50`）

| 项 | 内容 |
|----|------|
| **路径** | `data/kimi_mas_sft_minimax_minimal_native_train50/` |
| **Schema** | `minimal_contract_v1` |
| **规模** | 原始 **1456**（Main 397 + Sub 1059，17 tasks）；经预处理后 **1379** |
| **来源** | MiniMax 在 ScienceWorld train50 上做 native MAS rollout（Main 自己写 contract，Sub 自己 action） |
| **预处理** | → `..._prepared/`：6656 token fit、每 task cap 250 → train 908 / val 156 / test 315 |
| **特点** | 端到端 MAS 轨迹，非 gold replay；Main subgoal 由模型自己规划 |

**v3 更新（Kimi MAS v2）**：同目录重新上传 train_all 1456，split 调整后预处理仍为 1379 条，用于最新一轮 SFT。

---

### v4 · Expert Gold Replay（`expert_minimal_sft_train50`）

| 项 | 内容 |
|----|------|
| **路径** | 原始 `data/expert_minimal_sft_train50_success/` → 预处理 `..._prepared/` |
| **Schema** | `minimal_contract_v1` |
| **规模** | 原始 **3734**（50 ep，100% 成功）；预处理后 **1919**（train 1273 / val 148 / test 498） |
| **来源** | ScienceWorld **gold action 回放**：用专家动作重放成功轨迹，反推 Main contract |
| **Main 标签** | subgoal 常模板化，如 *"Execute the next expert step sequence: ..."* |
| **预处理** | 原始 71% 超 6656 → 预处理后 0% 超长；17 tasks |
| **特点** | 标签质量高（action 一定合法），但 Main contract 偏机械、泛化性存疑 |

---

### 数据对比小结

| 版本 | 协议 | 样本量 | Main 怎么来 | Sub 怎么来 | 任务覆盖 |
|------|------|--------|-------------|------------|----------|
| v1 | Full Contract | 4203 | Kimi 蒸馏 | 专家 action + native rollout | 23 |
| v1-clean | Full Contract | 4203 | v1 清洗 | 同 v1 | 23 |
| v2 | Minimal | 5691 | Kimi expert 抽取 | Kimi expert action | ~190 |
| v3 | Minimal | 1379 prep | MiniMax rollout | MiniMax rollout | 17 |
| v4 | Minimal | 1919 prep | Gold 反推 | Gold replay | 17 |

---

## 二、训练与评估结果

**统一说明**  
- **离线评测**：teacher-forcing，给 gold prompt，看标签复现率（valid / exact match）  
- **环境评测**：Stratified-145，`--protocol contract` 或 `minimal`，Main rep=1.2，Sub max_input=6656  
- **MGRPO**：在 SFT checkpoint 上做 RL，reward 来自环境 rollout

---

### v1 · Full Contract SFT

| | Main | Sub |
|---|------|-----|
| **训练** | 3 ep, lr 1e-4, max_len 1024 | 同左 |
| **val_loss** | 0.507 | 0.027 |
| **离线 exact** | 0%（valid 68%） | **80.7%**（native 子集仅 17.6%） |
| **Stratified-145** | — | mean **1.42**, success **0%**, valid 76%, **format_err 66%** |

---

### v1-clean · Main 重训

| | Main | Sub |
|---|------|-----|
| **训练** | 从 v1 Main 续训，lr 5e-5 | 仍用 v1 Sub |
| **val_loss** | 0.473 | — |
| **离线 exact** | 0%（valid **96%**） | — |
| **Stratified-145** | — | mean **5.47**, success **0%**, valid 83%, **format_err 0%** |

---

### v2 · Minimal Contract SFT

| | Main | Sub |
|---|------|-----|
| **训练** | 3 ep, lr 1e-4, max_len 1024 | 同左 |
| **离线 exact** | 0%（valid 98.5%） | **75.0%** |
| **Stratified-145** | 未正式跑 | — |
| **后续作用** | 作为 v3 SFT 的 warm-start | 同左 |

---

### v3 · Minimax Native SFT

| | Main | Sub |
|---|------|-----|
| **训练** | warm-start v2，lr 5e-5，max_len 4096，Main 8 ep / Sub 3 ep | 同左 |
| **val_loss** | **0.417** | **0.075** |
| **离线 exact** | 0%（valid 90%） | 0%* |
| **Stratified-145** | — | mean **6.07**, success **3.4%**, valid **85%**, fmt_err 39% |

\* Sub 离线 valid=0% 是评测格式问题，环境 valid 85% 说明 Sub 实际可用。

---

### v3-v2 · Kimi MAS v2 SFT（最新 native 线）

| | Main | Sub |
|---|------|-----|
| **训练** | warm-start v3，lr 2e-5，6656，Main 4 ep / Sub 3 ep 早停 | 同左 |
| **val_loss** | **0.426** | **0.077** |
| **离线 exact** | 0%（valid 96%） | **57.8%**（action 66%） |
| **Stratified-145** | 进行中 | — |

---

### v4 · Expert Gold SFT

| | Main | Sub |
|---|------|-----|
| **训练** | warm-start v3，6656；Main lr 2e-5 → val **0.131**；Sub lr 5e-6（2e-5 会 NaN） | Sub 训至 epoch 7/10 |
| **离线 exact** | **32%** | **96.7%** |
| **Stratified-145** | — | mean **2.57**, success **0%**, valid **52%**, fmt_err 31% |

环境评测用的是 Sub **epoch-1** checkpoint，非最终 best。

---

### MGRPO 结果

| 数据基座 | 训练线 | 状态 | 结果 |
|----------|--------|------|------|
| v1 Contract SFT | Sub-only | 9 iter 手动停 | 训练稳定，rollout 分无提升；未跑 stratified |
| v1 Contract SFT | Main-only | 20 iter | iter **8+** Main parse **0%**，空转 |
| v1 Contract SFT | Joint | 20 iter | iter **6+** 全线崩溃 |
| v3 Minimax SFT | Main-only v2 | 20 iter ✅ | iter20：kl≈0.31，Main format-invalid 40%；**未跑 stratified** |
| v3 Minimax SFT | Joint v2 | 1 iter | iter2 中断 |
| v3 Minimax SFT | Sub-only | — | bf16 OOM；4bit 训完 save 失败 |

---

### 环境评测总表（Stratified-145）

| 实验 | 数据 | Mean | Success | Action Valid | Format Err |
|------|------|------|---------|--------------|------------|
| Subtask RL 基线 | subtask 时代 | **18.12** | **9.0%** | 50% | 0% |
| Contract SFT v1 | v1 | 1.42 | 0% | 76% | **66%** |
| Contract mainclean | v1-clean | 5.47 | 0% | 83% | **0%** |
| Minimax native | v3 | **6.07** | **3.4%** | **85%** | 39% |
| Expert gold | v4 | 2.57 | 0% | 52% | 31% |
| Kimi MAS v2 | v3-v2 | TBD | TBD | TBD | TBD |

**Contract 时代环境最优 SFT**：v3 Minimax native（6.07 / 3.4%），仍远低于 subtask RL 基线。

---

## 三、问题分析

### 3.1 按版本看问题

**v1 Full Contract**  
- Main `action_guidance` 训练中出现重复退化（如同一 guidance 重复数十次）  
- max_length=1024 截断 JSON → 环境 **format error 66%**  
- Sub val_loss 极低（0.027）但 native_handoff 子集离线 exact 仅 **17.6%** — 两路数据 prompt 不一致，模型只学会了 contract_v2 分布  

**v1-clean**  
- 清洗 + rep1.2 + max_tokens=350 **彻底消除 format error**（66%→0%）  
- mean 从 1.42 升到 5.47，但 **success 仍 0%** — 格式问题解决后，瓶颈转为 Main 规划语义  

**v2 Minimal**  
- 协议简化后离线 Sub exact 75%，Main valid 98% 但 exact 仍 0%  
- 未跑环境评测；主要价值是作为 v3 warm-start  

**v3 Native Rollout**  
- **首个 contract 时代环境 success>0**（3.4%），Sub action valid 85%  
- Main exact 全线为 0 — Main 只会「格式合法」的 contract，不会复现 gold 语义  
- 数据仅 17 tasks，Stratified 30 tasks 中 **13 个 OOD**  

**v4 Expert Gold**  
- 离线 Sub **96.7%** vs 环境 valid **52%** — 最大「离线高、环境低」反差  
- 原因：(1) 离线给 gold contract，环境靠 Main 自己规划；(2) Main offline exact 仅 32%；(3) Gold Main subgoal 模板化，OOD task 泛化差  
- in-domain task mean ~+7，OOD task mean ~-3  

**MGRPO**  
- Main RL 在 iter 6–8 系统性崩溃：empty subgoal → 畸形 JSON → 350 tok 截断  
- 根因：SFT 起点太薄 + clip 过高 + 无 format gate / KL 约束  
- Sub-only MGRPO 6656 context 单卡不可行  

---

### 3.2 跨版本共性瓶颈

| 优先级 | 问题 | 证据 |
|--------|------|------|
| **P0** | **Main 规划是天花板** | 各线 success≈0；mainclean mean 仅 5.47；Expert Main exact 最高 32% |
| **P1** | **离线 Sub ≠ 环境 Sub** | Expert 97% offline vs 52% env-valid；环境 Main 写错 contract 则 Sub 再好也没用 |
| **P2** | **OOD 任务** | 训练 17 tasks vs 评测 30 tasks；OOD 拉低 mean |
| **P3** | **数据分布不一致** | v1 两路 Sub prompt 不同；v4 gold 模板化；v3 native 质量参差 |
| **P4** | **Main RL 不可直接训** | Contract MGRPO Main iter8+ 全崩 |

---

### 3.3 各干预的效果

| 干预 | 效果 |
|------|------|
| Main 标签清洗（v1→v1-clean） | format_err 66%→0%，mean 1.4→5.5 |
| 协议简化（v1→v2 minimal） | 离线 Sub 81%→75%（数据不同），但 prompt 与环境对齐 |
| Native rollout 数据（v3） | 首个 env success>0（3.4%） |
| Gold replay（v4） | 离线指标最高，但 env 反而低于 v3 |
| 6656 Sub prompt fit | 消除超长截断，Expert/Kimi v2 预处理 0% 超长 |
| Sub lr 5e-6（防 NaN） | Expert Sub 训练稳定 |

---

### 3.4 结论

Contract/minimal 全线 **尚未超越** subtask + Sub-only RL（18.12 / 9%）。  
当前最接近基线的是 **v3 Minimax native SFT**（6.07 / 3.4%），但 Main 规划弱、OOD 泛化差、离线指标不能代表环境表现，是后续数据与训练需要优先解决的三个方向。
