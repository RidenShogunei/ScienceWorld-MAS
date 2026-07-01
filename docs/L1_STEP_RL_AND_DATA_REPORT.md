# V7 之后 · 数据与 L1 单步 RL 实验报告

> 基座 `Qwen/Qwen3.5-9B` · minimal contract Main · action-id Sub（L1 / Plan A 线）  
> 日期：2026-06-30  
> 范围：V7 Contract SFT 之后新增的数据、SFT、以及 **L1 单步 GRPO** 全链路  
> **刻意不含**：L1 Main + V7 free-form Sub 的 145 局 episodic eval（训练/评估 Sub 协议不一致，见 §6）

---

## 一、结论（TL;DR）

| 阶段 | 核心指标 | 判定 |
|------|----------|------|
| **V7 SFT**（起点） | Stratified-145 mean **11.68** / success **2.8%** | contract 时代环境 eval 最优基线 |
| **L1 单步 · V7 Main 起点** | 71 states · expert_match **31.0%** · format **8.5%** | Main contract 几乎无用 |
| **L1 单步 · Main-only RL** | rollout best **49.2%** · greedy **46.5%** · format **~100%** | 格式与单步语义明显提升 |
| **L1 单步 · Joint RL** | rollout best **65.6%** @ iter 8 · sub_parse **100%** | **Main+Sub 联合更新显著优于只训 Main** |
| **Plan A SFT** | Main val **0.153** · Sub val **0.093**（epoch 7 early stop） | 简化 Main schema + action-id 管线跑通 |

**核心判断**

1. Contract 机制有效，但 V7 SFT 的 Main **单步 expert_match 仅 ~31%**，format 极差（~8%）。
2. L1 单步 RL 主要把 **format 拉满**，Main-only 把 expert_match 拉到 **~46–49%**。
3. **Joint 单步 RL**（同一 reward、同一 action-id Sub）把 expert_match 拉到 **65.6%**，说明 Sub 与 Main 需要一起优化。
4. 145 局 free-form Sub episodic eval **不能**代表 L1 训练效果；对齐的单步 greedy eval（71 states）才是可比的 offline 指标。

---

## 二、起点：V7 Expert-Subtask Contract SFT

| 项 | 内容 |
|----|------|
| 数据 | `data/expert_subtask_contract_v3_simple_minimax_sample1000/` |
| Checkpoint | `artifacts/checkpoints/sft_expert_subtask_contract_v3/` |
| Main 协议 | minimal contract JSON（subgoal / success_condition / …） |
| Sub 协议 | **free-form** `[action]…[/action][subtask_done]…` |
| 环境 eval | `artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json` |

| 指标 | 值 |
|------|-----|
| mean_score | **11.68** |
| success_rate | **2.76%** (4/145) |
| action_valid_rate | 16.8% |
| format_error_rate | 2.1% |
| mean_steps | 46.8 |

V7 上 **Sub-only / Joint MGRPO**（多步 episodic RL）未带来 greedy eval 净收益，已暂停；详见 [`artifacts/V7_CONTRACT_RL_REPORT.md`](../artifacts/V7_CONTRACT_RL_REPORT.md)。

本报告之后的工作转向：**固定 decision state 上的单步 GRPO** + **action-id Sub** + **Plan A 简化 Main**。

---

## 三、V7 之后新增数据

### 3.1 Action-ID SFT（L1 Sub 热启动）

| 项 | 内容 |
|----|------|
| 路径 | `data/action_id_sft_smoke/` |
| 任务 | find-plant · find-living-thing · power-component |
| 变体 | `no_contract` / `gold_contract` / `main_contract` |
| 规模 | 每变体 train **56** / val **16** |
| Sub 标签 | `selected_action_id: <int>`（从 ≤32 个候选中选） |
| 热启动 ckpt | `artifacts/checkpoints/action_id_sft_smoke/gold_contract/sub_agent/best` |

用途：L1 / Joint RL 的 **冻结或初始化 Sub**；候选动作来自 `valid_actions`，parse 成功后 **action_valid ≈ 100%**。

---

### 3.2 L1 Decision States（单步 RL 状态池）

| 项 | 内容 |
|----|------|
| 路径 | `artifacts/l1/decision_states_smoke.json` |
| 规模 | **71** states |
| 来源 | 3 tasks × 2 variations，gold replay 逐步截取 |
| 每 state 含 | observation · candidate_actions(≤32) · expert_action_id · gold_contract · recent_history |

配置见 `l1/config/smoke.yaml` / `joint.yaml`：`states_per_iter=32`，每 iter 随机采样。

---

### 3.3 Plan A SFT 数据（简化 Main + action-id Sub）

| 项 | 内容 |
|----|------|
| 路径 | `data/plan_a_sft_smoke/` |
| Schema | `plan_a_v1` — Main 只输出 `{subgoal, focus_objects}` |
| 规模 | train **2368** / val **238** / test **303**（合计 **2909**） |
| 构成 | Main **500** + Sub **2409** |
| 来源 | multisquare expert subtask 轨迹 |
| 生成 | `plan_a/generate_sft_data.py` |

Manifest：`data/plan_a_sft_smoke/manifest.json`

---

## 四、Plan A SFT 训练结果

脚本：`scripts/run_plan_a_sft_smoke.sh` · GPU 4  
Checkpoint：`artifacts/checkpoints/plan_a_sft_smoke/`

| Agent | Epochs | best val_loss | 备注 |
|-------|--------|---------------|------|
| **Main** | 10/10 | **0.153** | 1.42 → 0.15，无 early stop |
| **Sub** | 7/10 | **0.093** | epoch 4 最佳，patience=3 early stop |

Sub warm-start：`action_id_sft_smoke/gold_contract/sub_agent/best`

Plan A 尚未做 RL 或环境 eval；与 L1 并行，验证「更简 Main schema + 同一 action-id Sub」的 SFT 可行性。

---

## 五、L1 单步 GRPO 框架

| 组件 | 说明 |
|------|------|
| 代码 | `l1/trainer.py` · `l1/rollout.py` · `l1/reward.py` |
| 算法 | 组内相对 GRPO（G=4 samples/state） |
| Rollout | Main sample contract → Sub **action-id**（训练时可 sample）→ env 探针一步 |
| Reward | expert_match **1.0** + action_valid **0.2** + format_valid **0.1** − format_penalty |
| Main 热启动 | V7 SFT → smoke；continue / joint 从 RL checkpoint 续 |
| Sub（Main-only 线） | 冻结 `action_id_sft_smoke/gold_contract/sub_agent/best` |
| Sub（Joint 线） | 与 Main **同时更新**，`sub_lr=2e-5` > `main lr=1e-5` |

`train.agents`：`main`（仅 Main）| `both`（Main + Sub 联合）。

---

## 六、刻意未收录的 Eval

以下结果 **不写入本报告主表**，因训练与评估 **Sub 协议不一致**：

| Eval | 配置 | 问题 |
|------|------|------|
| `l1_main_step_rl_smoke_iter0010_stratified145` | L1 Main + **V7 free-form Sub** | 训练用 action-id，eval 用 `[action]` 自由生成 |
| 同上 | action_valid 仅 ~20% | 主要来自 free-form Sub，不能反映选择题 Sub |

该 eval 结果（success **1.38%** / mean **10.69**）**低于** V7 baseline，但 **不能**作为 L1 RL 有效性的结论依据。

后续已新增 **`--agent-interface action-id`** episodic eval（`scripts/run_eval_l1_joint_action_id_stratified145.sh`）；跑完后应单独成表，不与上表混读。

---

## 七、L1 Main-only 单步 RL

### 7.1 Smoke（iter 1–10）

| 项 | 内容 |
|----|------|
| 配置 | `l1/config/smoke.yaml` |
| 脚本 | `scripts/l1_smoke.sh` |
| 日志 | `artifacts/l1/l1_smoke_train.log` |
| Checkpoint | `artifacts/checkpoints/l1_main_step_rl_smoke/iter_0001` … `iter_0010` |
| Main 初始化 | V7 SFT `main_agent/best` |
| Sub | 冻结 action-id SFT |

**每 iter rollout expert_match（%）**

| iter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|------|---|---|---|---|---|---|---|---|---|---|
| expert | 31.2 | 34.4 | 21.1 | 33.6 | 35.2 | **49.2** | 38.3 | 40.6 | **49.2** | 48.4 |

- iter 6 / 9 峰值 **49.2%**
- format_valid 从 ~8% 升至 **~97–100%**（smoke 日志无逐 iter format 字段；greedy eval 见 §9）

### 7.2 Continue + Early Stop（iter 11–14）

| 项 | 内容 |
|----|------|
| 配置 | `l1/config/continue.yaml`（max 60 iter，patience=3） |
| 脚本 | `scripts/l1_continue.sh` |
| 日志 | `artifacts/l1/l1_continue_train.log` |
| 续训起点 | `iter_0010/main` |

| iter | expert_match | format_valid |
|------|--------------|--------------|
| 11 | **46.1%** | 100.0% |
| 12 | 39.1% | 98.4% |
| 13 | 44.5% | 99.2% |
| 14 | 43.8% | 98.4% |

**Early stop @ iter 14** · best **iter_0011**（expert **46.1%**）  
未超过 smoke 峰值 49.2%（iter 6/9）。

---

## 八、L1 Joint 单步 RL（Main + Sub）

| 项 | 内容 |
|----|------|
| 配置 | `l1/config/joint.yaml` |
| 脚本 | `scripts/l1_joint.sh` |
| 日志 | `artifacts/l1/l1_joint_train.log` |
| Checkpoint | `artifacts/checkpoints/l1_joint_step_rl_smoke/` |
| Main 初始化 | `l1_main_step_rl_smoke/iter_0011/main` |
| Sub 初始化 | `action_id_sft_smoke/gold_contract/sub_agent/best` |
| Rollout | Main sample + **Sub sample** |

**每 iter rollout expert_match（%）**

| iter | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|------|---|---|---|---|---|---|---|---|---|---|-----|
| expert | 42.2 | 45.3 | 45.3 | 51.6 | 57.8 | 45.3 | 46.9 | **65.6** | **65.6** | 64.8 | 53.1 |

| iter | format_valid | sub_parse |
|------|--------------|-----------|
| 全程 | **97–100%** | **100%** |

**Early stop @ iter 11** · best **`iter_0008`** · expert_match **65.6%**

### Main-only vs Joint（单步 rollout）

| 方法 | best expert_match | format | Sub 是否更新 |
|------|-------------------|--------|--------------|
| V7 SFT（未 RL） | 31.0%（greedy 71） | 8.5% | — |
| Main-only RL | 49.2%（rollout）/ 46.5%（greedy） | ~100% | 否 |
| **Joint RL** | **65.6%** | ~100% | **是** |

Joint 比 Main-only **+16~19 pp** expert_match，是本轮最显著增益。

---

## 九、单步 Greedy Eval（71 fixed states，协议对齐）

与 L1 训练 **同一协议**：minimal Main + action-id Sub + 相同 71 states。  
脚本：`scripts/l1_eval.sh` / `l1/eval.py`

| Checkpoint | expert_match | format_valid | action_valid | parse_success | mean_reward_delta |
|------------|--------------|--------------|--------------|---------------|-------------------|
| **V7 SFT Main**（`eval_baseline.json`） | **31.0%** | **8.5%** | 88.7% | 100% | −5.70 |
| **Main-only iter_0010**（`eval_step_rl.json`） | **46.5%** | **100%** | 93.0% | 100% | +5.94 |

说明：

- **expert_match** = Sub 所选 action_id 是否等于 gold expert_action_id（主优化目标）
- **format_valid** = Main contract JSON 是否可 parse
- **action_valid** = 所选动作是否在环境 `valid_actions` 中；action-id 且 parse 成功时通常很高
- 此 eval **不含** 145 局多步 score / success

---

## 十、指标释义（避免混读）

| 指标 | 含义 | L1 单步典型值 |
|------|------|---------------|
| expert_match | Sub 是否选对 expert 动作 | 31% → 66% |
| format_valid | Main JSON 是否合法 | 8% → 100% |
| sub_parse | Sub 是否输出合法 action_id | Joint 全程 100% |
| action_valid | 环境是否接受该 action 字符串 | action-id 下 ~90%+ |
| mean_score / success | **多步 episodic** 任务完成度 | 见 V7 baseline；**本报告不收录 L1 错配 eval** |

---

## 十一、产物路径索引

| 用途 | 路径 |
|------|------|
| V7 SFT eval | `artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json` |
| L1 states | `artifacts/l1/decision_states_smoke.json` |
| 单步 greedy eval | `artifacts/l1/eval_baseline.json` · `artifacts/l1/eval_step_rl.json` |
| Main-only ckpt | `artifacts/checkpoints/l1_main_step_rl_smoke/iter_0011/main`（continue best） |
| Joint ckpt | `artifacts/checkpoints/l1_joint_step_rl_smoke/iter_0008/`（main + sub） |
| Plan A ckpt | `artifacts/checkpoints/plan_a_sft_smoke/` |
| Plan A 数据 | `data/plan_a_sft_smoke/` |
| Action-id 数据 | `data/action_id_sft_smoke/` |

---

## 十二、下一步建议

1. **Joint iter_0008** 跑 **action-id 对齐** 的 145 局 eval（脚本已就绪），与 V7 比 mean/success。
2. **Plan A** 在 71 states 或 stratified-145 上跑 smoke eval（Main+Sub 均用 plan_a ckpt）。
3. 若 Joint episodic 仍不涨：瓶颈可能在 **每步 Main replan** vs 训练时 chunk contract，或 task 覆盖（71 states ⊂ 3 tasks）。
4. 继续扩大 decision states（更多 task/variation）再训 Joint，观察 expert_match 是否可稳定 >65%。

---

## 附录：与旧报告关系

- Contract 全版本数据对比：[`artifacts/RECENT_DATA_SFT_REPORT.md`](../artifacts/RECENT_DATA_SFT_REPORT.md)
- V7 多步 MGRPO：[`artifacts/V7_CONTRACT_RL_REPORT.md`](../artifacts/V7_CONTRACT_RL_REPORT.md)
- **本报告**：V7 之后 → Plan A 数据/SFT → L1 单步 Main-only / Joint RL → 对齐的单步 greedy eval
