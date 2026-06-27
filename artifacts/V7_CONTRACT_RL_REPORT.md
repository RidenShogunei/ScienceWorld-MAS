# V7 Expert-Subtask Contract · MGRPO 实验总结

> 基座 `Qwen/Qwen3.5-9B` · minimal protocol · Stratified-145（145 ep，30 task，greedy fp16 eval）  
> SFT 数据：`data/expert_subtask_contract_sft_v3_simple_minimax_sample1000/`  
> 日期：2026-06

---

## 一、结论（TL;DR）

**Contract/minimal 协议下，Sub-only 与 Joint MGRPO 均未在 Stratified-145 greedy eval 上带来可验证收益。**

| 阶段 | Stratified-145 mean | success | 判定 |
|------|---------------------|---------|------|
| **V7 SFT**（主线） | **11.68** | **2.8%** (4/145) | 当前最优 checkpoint |
| Sub-only MGRPO v3 iter_2 | 11.66 | 2.8% (4/145) | ≈ SFT |
| Sub-only MGRPO v3 iter_10 | 11.68 | 2.1% (3/145) | ≈ SFT，success 略降 |
| Joint MGRPO v1 iter_10/12 | — | — | eval **33/145 中断**；partial mean ~10.7–11.0，0 success |

Pipeline **能跑通**（rollout、梯度、checkpoint 均有效），但 **优化目标与 greedy eval 不对齐**，KL 把策略拴在 SFT 附近，greedy 行为几乎不变。

**建议：** 以 V7 SFT 为部署/后续实验基线；Contract RL 暂停，优先数据与 Main 规划，而非继续拧 MGRPO 超参。

---

## 二、V7 SFT 基线

| 项 | 内容 |
|----|------|
| 数据 | Expert gold replay + MiniMax contract，`minimal_contract_v1` |
| Checkpoint | `artifacts/checkpoints/sft_expert_subtask_contract_v3/` |
| Eval | `artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json` |

| 指标 | 值 |
|------|-----|
| mean_score | 11.68 |
| success_rate | 2.8% (4/145) |
| action_valid_rate | 16.8% |
| format_error_rate | 2.1% |
| mean_steps | 46.8 |

对照：subtask 时代 Sub-only RL 基线 mean **18.12** / success **9.0%**（不同协议，不可直接类比为「RL 一定该更高」）。

---

## 三、Sub-only MGRPO（v2 → v3）

### 3.1 v2：Pipeline 致命 Bug（训练无效）

| Bug | 现象 | 修复 |
|-----|------|------|
| `minimal` 误用 verbose parser | Main parse ~0%，Sub 从未调用 | `agent_protocol.py` → `parse_minimal_contract_response` |
| Rollout 4bit 推 Main | JSON parse ~28% | rollout 改 **fp16**（`--no-rollout-use-4bit`） |
| 全 greedy rollout | 组内轨迹相同，advantage=0 | Sub **sample**（T=0.7） |

v2 约 20 iter **零有效梯度**；修复后 v3 每 iter 32/32 有效 rollout，Sub 800+ samples/iter，loss 下降，权重变化（SFT→iter10 L2≈0.078）。

### 3.2 v3：Eval 仍 ≈ SFT

脚本：`scripts/run_sub_only_mgrpo_expert_subtask_contract_v3.sh`

| 指标 | SFT | RL iter_2 | RL iter_10 |
|------|-----|-----------|------------|
| mean | 11.68 | 11.66 | 11.68 |
| success | 4/145 | 4/145 | **3/145** |
| action_valid | 16.8% | 15.2% | 17.1% |

- **121/145** ep greedy 分数与 SFT **完全相同**
- 12 ep 变好、7 ep 变差，净效果 ≈ 0

### 3.3 Sub-only 无效根因

1. **只训 Sub，Main plan 锁死** — 瓶颈在 contract/子目标，Sub 微调空间有限  
2. **GRPO 组内相对 reward** — 优化「同题 4 条里谁更好」，不是固定 145 题的 success  
3. **KL（beta=0.05）** — 显式限制偏离 SFT  
4. **Train sample / Eval greedy** — RL 在采样路径上学，greedy eval 看不到  
5. **每 iter 随机 dev 8 题** — 与 Stratified-145 分布不一致  

---

## 四、Joint MGRPO（Main + Sub）

### 4.1 动机

Sub-only 推不动 eval → 尝试 **Joint**：Main 与 Sub 同时从环境反馈更新，让 plan 也能变。

脚本：`scripts/run_joint_mgrpo_expert_subtask_contract_v3.sh`

| 配置 | 值 |
|------|-----|
| agents | both |
| rollout 池 | Stratified-145（`--episode-list`） |
| 每 iter | 8 组 × 4 = 32 rollout |
| rollout | fp16；Main **greedy**，Sub **sample** T=0.7 |
| lr | Main 2e-6，Sub 1e-5 |
| 护栏 | strict-format-gate，beta=0.03，clip 0.1 |

### 4.2 训练健康度（iter 1–12，iter 13 手动停止）

| 观察 | 结果 |
|------|------|
| Main format 崩溃 | **未发生**（format 无效 0–5/iter，main_fmt 96–100%） |
| 组内 advantage | 8/8 组有方差（adv_std≈1） |
| Rollout 采样 success | iter_9 最好：mean 20.6，**4× score=100** |
| neg100 | iter 间波动大（0–11/32），属采样方差 |

Checkpoint 保留至 `iter_0012`：`artifacts/checkpoints/mgrpo_expert_subtask_contract_v3_joint_v1/`（本地，gitignore）。

### 4.3 Eval（中断）

iter_10 / iter_12 于 GPU 1/2 启动 Stratified-145 greedy eval，**33/145 时手动停止**（无完整 JSON）。

| | partial (33 ep) | SFT (145 ep) |
|--|-----------------|--------------|
| mean | ~10.7–11.0 | 11.68 |
| success | 0/33 | 4/145 (2.8%) |

早期信号 **不优于 SFT**；完整 145 题未跑完，不宜作最终结论，但结合 Sub-only 结果，**继续 joint RL 优先级低**。

### 4.4 「训练有 100 分、eval 不涨」的原因

```
训练 rollout：Sub sample → 组内出现不同轨迹，偶发 100 分
Stratified eval：Sub greedy  → 行为贴近 SFT，看不到采样路径上的改进
```

加上 KL + 低 Main lr，**权重在变，greedy 策略几乎不变**（Sub-only 已证 121/145 ep 同分）。

---

## 五、ScienceWorld 分数与指标

### 5.1 final_score = -100

ScienceWorld API：内部 score < 0 时映射为 **-100**，**episode 立即结束**。

常见触发：**选错对象**（如对 living 物体 focus 当 non-living）、**错误 declare complete** 等致命步骤。  
不是 JSON 格式错误；与 `action_valid=false` 常同时出现。

Reward 里 `_normalized_score` 把负分 clamp 到 0，故 -100 与 0 在 global_score 项上等价；组内比较时仍会因大幅落后其他 rollout 而得到负 advantage。

### 5.2 建议监控的指标

| 层级 | 指标 | 用途 |
|------|------|------|
| **Eval（主）** | mean_score, success_rate | 与 SFT 对比的唯一可靠标准 |
| Eval | action_valid_rate, format_error_rate | 动作/格式健康 |
| Rollout（辅） | main_format_rate, neg100 次数 | 防 Main 崩 |
| 训练（辅） | kl, clip, adv_std | 梯度是否有效 |
| Rollout mean | 噪声大 | **勿当主指标** |

每 iter 自动汇总：`iter_XXXX/rollouts.json`（本地 checkpoint 目录）。

---

## 六、代码与脚本变更（可复现）

| 文件 | 说明 |
|------|------|
| `agent_protocol.py` | minimal parse 修复 |
| `mgrpo_trainer.py` | `--episode-list`；contract rollout 默认 fp16 |
| `scripts/run_sub_only_mgrpo_expert_subtask_contract_v3.sh` | Sub-only v3 |
| `scripts/run_joint_mgrpo_expert_subtask_contract_v3.sh` | Joint v1 |
| `scripts/run_eval_joint_mgrpo_expert_subtask_contract_v3.sh` | Joint eval |

**注意：** bash 内置变量 `GROUPS` 会与 `--groups` 冲突；joint 脚本已改为 `MGRPO_GROUPS`。

---

## 七、若未来仍做 Contract RL

按优先级：

1. **Reward 加 success bonus** + 固定 Stratified-145 全量或高覆盖 rollout  
2. **弱 KL / anneal** + 适当提高 Main lr（配合 strict-format-gate）  
3. **Eval 加 best-of-N sample** 测策略上限；或 **greedy rollout** 对齐 train/eval  
4. **先换 SFT 数据**（如 v3 native，env success 更高）再 RL — 起点太低时 RL 救不了 Main  

---

## 八、Artifacts 索引

| 路径 | 内容 |
|------|------|
| `artifacts/eval/sft_expert_subtask_contract_v3_stratified145.json` | V7 SFT 完整 eval |
| `artifacts/eval/mgrpo_expert_subtask_contract_v3_v3_iter02_stratified145.json` | Sub-only iter_2 |
| `artifacts/eval/mgrpo_expert_subtask_contract_v3_v3_iter10_stratified145.json` | Sub-only iter_10 |
| `artifacts/eval/mgrpo_joint_v1_iter10_stratified145.log` | Joint partial eval（33/145） |
| `artifacts/RECENT_DATA_SFT_REPORT.md` | Contract 时代数据与 SFT 全史 |

---

## 九、最终建议

1. **生产/对比基线：** `artifacts/checkpoints/sft_expert_subtask_contract_v3/`  
2. **Contract MGRPO：** 暂停；v2 bug 已修、v3/joint pipeline 健康，但 **greedy eval 无净收益**  
3. **下一步：** Main 规划与 SFT 数据（OOD、native rollout）优先于 RL 微调  
