# ScienceWorld-MAS 完整实验报告

**生成日期**：2026-06-15  
**代码基线**：Git commit `4bff7f0133deff49e3045fa796d0a4b071fd05ec`  
**实验周期**：2026-06-12 ~ 2026-06-15  

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [实验环境](#2-实验环境)
3. [任务、数据与系统架构](#3-任务数据与系统架构)
4. [训练设计](#4-训练设计)
5. [评测协议](#5-评测协议)
6. [实验结果](#6-实验结果)
7. [数据分析](#7-数据分析)
8. [实现问题与修复](#8-实现问题与修复)
9. [结论与建议](#9-结论与建议)
10. [产物索引与复现命令](#10-产物索引与复现命令)

---

## 1. 执行摘要

本阶段在 ScienceWorld 上构建并评估 **分层 Main/Sub Agent**（Main 规划 subtask，Sub 执行 low-level action），完成了四条实验线：

| 实验线 | 简述 | 环境侧结论 |
|--------|------|-----------|
| **SFT 基线** | 从 Multi-Square 专家轨迹训练 Main/Sub LoRA | dev-20 均分 9.2，action valid 23.4% |
| **Sub 续训 SFT（sft_sub_v2）** | 从 SFT Sub 续训 1 epoch | dev-20 均分 **16.0**，valid **35.3%**（dev-20 最优） |
| **Joint M-GRPO（mgrpo_v2）** | Main+Sub 联合 RL，13/20 iter 后停止 | dev-20 均分 14.1，valid 22.3%（未解决 Sub 瓶颈） |
| **Sub-only M-GRPO（mgrpo_sub_only）** | 冻结 Main，仅 RL 更新 Sub；iter_7 后停止 | **stratified-145 全面最优**：均分 **18.1**，valid **50.3%**，成功率 **9.0%** |

**核心结论**：

1. **Sub 执行器是系统瓶颈**，Main 离线极强（exact match 98.4%），但环境 action valid 长期停留在 23–35%（SFT/续训阶段）。
2. **Offline 高 exact match 不能预测环境表现**——根因是 hierarchical 部署时的 **distribution shift**（Main 偏离 + 环境反馈偏离专家轨迹）。
3. **Joint M-GRPO 机制修复后可用**，但对固定 dev 评测提升有限，且 action valid 未改善。
4. **Sub-only RL（iter_7）是当前最佳方案**：在更可信的分层评测（145 ep，30 task 全覆盖）上，均分、成功率、action valid 三项均领先；action valid 从 SFT ~28% 提升至 **50%**。
5. **距离可用 agent 仍有明显差距**：145 ep 成功率仅 9%，测量/物化类 task 仍大量失败。

**推荐部署 checkpoint**：

- Main：`artifacts/checkpoints/sft/main_agent/best`（全程固定）
- Sub：`artifacts/checkpoints/mgrpo_sub_only/iter_0007/sub`

---

## 2. 实验环境

### 2.1 硬件

| 项目 | 配置 |
|------|------|
| 操作系统 | Linux 6.8.0（Ubuntu 系） |
| GPU | 多卡 NVIDIA（训练/评测使用单卡 `CUDA_VISIBLE_DEVICES`） |
| 典型显存占用 | bf16 全精度推理/训练 ~17–26 GB（Qwen3.5-9B + 双 LoRA） |

### 2.2 软件栈

| 组件 | 版本 |
|------|------|
| Python | 3.10 / 3.11 |
| Java (JRE) | `/home/jinxu/jdk-21.0.11+10-jre`（ScienceWorld 必需） |
| PyTorch | 2.12.0 |
| Transformers | 5.9.0 |
| PEFT | 0.19.1 |
| bitsandbytes | 0.49.2 |
| ScienceWorld | 1.2.3 |
| HuggingFace 镜像 | `HF_ENDPOINT=https://hf-mirror.com`（可选） |

### 2.3 基础模型与适配器

| 项目 | 配置 |
|------|------|
| Base Model | `Qwen/Qwen3.5-9B` |
| 精度 | bf16 LoRA（训练与正式评测均 `--no-use-4bit`） |
| LoRA | r=16, alpha=32, dropout=0.05，目标模块 q/k/v/o_proj |
| 可训练参数量（Sub adapter） | ~393 万（Sub-only RL 日志） |

### 2.4 环境变量（典型）

```bash
export JAVA_HOME=/home/jinxu/jdk-21.0.11+10-jre
export PATH=$JAVA_HOME/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com   # 可选
export CUDA_VISIBLE_DEVICES=0              # 按卡分配
```

---

## 3. 任务、数据与系统架构

### 3.1 ScienceWorld 任务规模

| Split | Variation 总数 | Task type 数 |
|-------|----------------|--------------|
| train | ~14,000+ | 30 |
| dev | **1,796** | 30 |
| test | **1,819** | 30 |

每个 variation 对应一个 `(task_name, variation_id)` episode，环境 step_limit=50，max_subtasks=15。

### 3.2 数据来源

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 原始数据 | Multi-Square 专家轨迹 | 经 `prepare_multisquare.py` 转换 |
| 审计 | `audit_multisquare.py` | 输出 `artifacts/data_audit.json`，无 malformed 样本 |
| SFT 样本 | `generate_sft_data.py` | 生成 Main/Sub 对话格式 |

**SFT 数据量**：

| Agent | Train | Val |
|-------|-------|-----|
| Main | 11,521 | 984 |
| Sub | 54,691 | 3,351 |

专家数据特点：**仅正轨迹、按 subtask 切片**，缺少 invalid-action 负样本，也缺少 Main 带偏后的状态覆盖。

### 3.3 分层架构

```text
ScienceWorld Environment
        ↑↓ observation / reward / score
    Sub Agent（执行器）
        - 输入：(subtask, observation)
        - 输出：[action]...[/action][subtask_done]true|false[/subtask_done]
        ↑ subtask
    Main Agent（规划器）
        - 输入：(task, observation, 已完成 group actions)
        - 输出：[subtask]...[/subtask]
```

- **Main**：负责分解 subtask，不做 low-level action。
- **Sub**：在 Main 给定 subtask 下逐步与环境交互。
- 解码：环境评测与 rollout 默认 **贪心**（`do_sample=False`）；MGRPO rollout 使用 **随机采样**（temperature=0.7/0.9）。

### 3.4 M-GRPO 设计要点

详见 `docs/MGRPO_DESIGN.md`：

- **Group 定义**：同一 `(split, task_name, variation_id)` 的 G 条完整 rollout 为一组。
- **Group advantage**：组内 reward 标准化，\(A = (R - \mu) / (\sigma + \epsilon)\)。
- **Sub 对齐**：每条 rollout 映射到固定 M 个 Sub-invocation slot，Sub advantage **独立**于 Main 计算。
- **Reward 组成**（默认权重）：global_score 0.5、progress 0.3、format 0.1、action_validity 0.1，及 no_progress/repetition 惩罚。

---

## 4. 训练设计

### 4.1 实验一：SFT 基线

| 项目 | 配置 |
|------|------|
| 脚本 | `sft_trainer.py` |
| Agents | Main + Sub 分别训练 |
| Main | 2 epoch，best val loss **0.0059** |
| Sub | ~1 epoch（epoch 2 未完整），epoch 1 val loss **0.1489** |
| 输出 | `artifacts/checkpoints/sft/main_agent/best`，`sub_agent/best` |

### 4.2 实验二：Sub 续训 SFT（sft_sub_v2）

| 项目 | 配置 |
|------|------|
| 初始化 | `--init-adapter` 从 `sft/sub_agent/best` |
| 学习率 | 1e-4 |
| 计划 / 实际 | 3 epoch 计划，**完成 epoch 1 后停止** |
| 输出 | `artifacts/checkpoints/sft_sub_v2/sub_agent/best` |

| 指标 | 初次 SFT ep1 | 续训 ep1 |
|------|-------------|----------|
| train_loss | 0.0599 | **0.0256** |
| val_loss | 0.1489 | 0.1575（略差） |

### 4.3 实验三：Joint M-GRPO（mgrpo_v2）

| 项目 | 配置 |
|------|------|
| 脚本 | `mgrpo_trainer.py` |
| 初始化 | SFT best Main + Sub |
| Agents | `--agents both`（Main + Sub 联合更新） |
| 采样 | 8 groups × 4 rollouts = **32 ep/iter** |
| 计划 | 20 iter，**实际 13/20 后停止** |
| 学习率 | 1e-4 |
| Rollout | temperature=0.7，随机采样 |
| 输出 | `artifacts/checkpoints/mgrpo_v2/iter_0001` … `iter_0013` |

**训练 batch 统计（13 iter × 32 = 416 rollouts）**：

| 指标 | 值 |
|------|-----|
| 整体 batch 均分 | 16.1 |
| 满分轨迹 | 20/416 |
| iter_7 batch 峰值 | **36.9** |
| iter_13 batch | 4.7（退化） |

### 4.4 实验四：Sub-only M-GRPO（mgrpo_sub_only）

| 项目 | 配置 |
|------|------|
| 脚本 | `scripts/run_sub_only_mgrpo.sh` |
| Main | **冻结** `sft/main_agent/best` |
| Sub 初始化 | `sft_sub_v2/sub_agent/best` |
| Agents | `--agents sub` |
| 采样 | 8 groups × 4 = 32 ep/iter |
| 计划 | 20 iter，**实际 12/20 后手动停止**（iter_13 进行中停止） |
| Rollout temperature | **0.9** |
| Reward | `--reward-action-validity 0.3`（默认 0.1 的 3 倍） |
| 输出 | `artifacts/checkpoints/mgrpo_sub_only/iter_0001` … `iter_0012` |

**逐 iter 训练 batch 均分**：

| Iter | Batch 均分 | 满分/32 | 无效(~-100) | loss | kl |
|------|-----------|---------|-------------|------|-----|
| 1 | -12.6 | 1 | 10 | -0.047 | 0.0016 |
| 6 | 38.3 | 11 | 3 | 0.224 | 0.0003 |
| **7（峰值）** | **54.2** | 9 | 2 | 0.150 | 0.0003 |
| 8 | -2.8 | 6 | 11 | 0.333 | 0.0002 |
| 9 | -15.6 | 1 | 9 | -0.119 | 0.0005 |
| 10 | 9.2 | 1 | 2 | -0.098 | 0.0017 |
| 11 | -13.2 | 0 | 6 | -0.257 | 0.0004 |
| 12 | 2.2 | 2 | 7 | 0.189 | 0.0003 |

iter 7 之后 batch 明显震荡回落，故在 iter_13 进行中停止；**iter_7 为 Sub-only RL 最佳 checkpoint**。

---

## 5. 评测协议

本阶段使用了 **三类评测**，重要性不同：

### 5.1 离线评测（`evaluate_sft.py`）

| 项目 | 配置 |
|------|------|
| 数据 | val split，n=500 |
| 指标 | format valid、exact match、action_exact、done_accuracy |
| 用途 | 诊断格式与 memorization，**不能替代环境评测** |

### 5.2 快速环境评测：dev-20

| 项目 | 配置 |
|------|------|
| 脚本 | `evaluate_environment.py` |
| Split | dev |
| Episodes | **20**（seed=123 随机 shuffle 后取前 20） |
| 覆盖 | 仅 **13/30** task type |
| 解码 | 贪心，step_limit=50 |
| 用途 | 训练 iter 间快速对照（**方差大**） |

### 5.3 正式环境评测：dev stratified-145（方案 A）

| 项目 | 配置 |
|------|------|
| 协议 | 每个 task type 固定抽 **K=5** 条 variation（seed=123） |
| 列表文件 | `artifacts/eval/dev_stratified_k5_seed123.json` |
| Episodes | **145**（30 task 全覆盖；2 个 task dev variation <5，各取 2 条） |
| 脚本 | `scripts/run_stratified_eval.sh` |
| 用途 | **阶段决策与横向对比的正式指标** |

> **注意**：dev-20 与 stratified-145 的数字不可直接对比；后者覆盖更全、更可信。

---

## 6. 实验结果

### 6.1 离线评测（val, n=500）

| Agent | Format Valid | Exact Match | Action Exact | Done Acc |
|-------|-------------|-------------|--------------|----------|
| Main (SFT) | 100% | **98.4%** | — | — |
| Sub (SFT) | 100% | 77.0% | 80.6% | 81.8% |
| Sub v2 | 100% | 78.6% | **88.6%** | 83.4% |

### 6.2 环境评测：dev-20（seed=123）

| 方法 | Main | Sub | 均分 ↑ | 成功率 | Action Valid ↑ | Format Error |
|------|------|-----|--------|--------|----------------|--------------|
| SFT 基线 | sft/best | sft/best | 9.2 | 0% | 23.4% | 0% |
| SFT + Sub v2 | sft/best | sub_v2/best | **16.0** | 0% | **35.3%** | 0% |
| Joint MGRPO iter_7 | mgrpo_v2/iter_7 | mgrpo_v2/iter_7 | 14.1 | 0% | 22.3% | 0% |
| Joint MGRPO iter_13 | mgrpo_v2/iter_13 | mgrpo_v2/iter_13 | 12.4 | 0% | 18.4% | 0% |
| Sub-only RL iter_7 | sft/best | mgrpo_sub_only/iter_7/sub | 14.3 | **5%** (1/20) | 38.9% | 0% |

### 6.3 环境评测：dev stratified-145（正式）

| 方法 | 均分 ↑ | 成功率 ↑ | Action Valid ↑ | 平均步数 |
|------|--------|----------|----------------|----------|
| SFT + Sub v2 | 11.8 | 4.1% (6/145) | 28.4% | 44.9 |
| Joint MGRPO iter_7 | 11.1 | 2.8% (4/145) | 14.8% | 47.5 |
| **Sub-only RL iter_7** | **18.1** | **9.0% (13/145)** | **50.3%** | **42.1** |

**结果文件**：

- `artifacts/eval/sft_sub_v2_stratified145.json`
- `artifacts/eval/mgrpo_v2_iter07_stratified145.json`
- `artifacts/eval/mgrpo_sub_only_iter07_stratified145.json`

### 6.4 Sub-only RL iter_7 满分 episode（13/145）

| Task | Variation | Action Valid |
|------|-----------|--------------|
| chemistry-mix | 19 | 67% |
| chemistry-mix-paint-secondary-color | 23, 24 | 59%, 44% |
| find-animal | 195, 196 | 75%, 77% |
| find-non-living-thing | 211, 219 | 100%, 78% |
| inclined-plane-determine-angle | 92 | 67% |
| inclined-plane-friction-unnamed-surfaces | 117 | 60% |
| lifespan-longest-lived | 89 | 50% |
| mendelian-genetics-known-plant | 80 | 61% |
| power-component | 10, 11 | 69%, 79% |

---

## 7. 数据分析

### 7.1 Offline ≠ Environment：Distribution Shift

```text
专家 SFT 数据：(subtask_expert, obs_expert) → action_expert
分层部署：    (subtask_main,  obs_env)     → action_?
```

- Main 离线 98.4% exact，但环境中 Sub 的 obs 迅速偏离专家路径。
- Sub v2 离线 action_exact 88.6%，环境 valid 仅 28–35%（dev-20/stratified）。
- **结论**：应优先优化 **环境反馈下的 Sub 鲁棒性**，而非继续刷 offline val loss。

### 7.2 dev-20 vs stratified-145：评测协议影响结论

| 方法 | dev-20 均分 | stratified 均分 | dev-20 valid | stratified valid |
|------|------------|-----------------|--------------|------------------|
| Sub v2 | **16.0** | 11.8 | 35.3% | 28.4% |
| Sub-only RL | 14.3 | **18.1** | 38.9% | **50.3%** |

- Sub v2 在 dev-20 上更好，但在 stratified-145 上更差 → **dev-20 对 Sub v2 偏友好**（20 条、13 task，方差大）。
- Sub-only RL 在更正规评测上全面领先 → **Sub-only RL 的真实收益被 dev-20 低估**。
- **教训**：不应以 training batch 均分或 dev-20 作为唯一决策依据。

### 7.3 同 145 条 Head-to-Head

| 对比 | 更好 | 更差 | 相同 |
|------|------|------|------|
| Sub-only RL vs Sub v2 | **48** | 24 | 73 |
| Sub-only RL vs Joint | **77** | 27 | 41 |
| Sub v2 vs Joint | 42 | 31 | 72 |

独占最高分 episode 数：Sub-only RL **46**，Joint 16，Sub v2 14。

### 7.4 按 Task 类别均分（stratified-145）

| 类别 | Sub v2 | Joint MGRPO | Sub-only RL |
|------|--------|-------------|-------------|
| find（找物体/生物） | 27.8 | 23.1 | **30.6** |
| biology（生命周期/遗传） | 23.6 | 11.9 | **26.0** |
| incline（斜面） | 11.0 | 5.0 | **13.3** |
| chem/phase（化/相变） | 4.0 | 7.7 | 5.7 |
| measurement（测量/导电/温度计） | **-1.3** | 14.5 | 5.6 |
| other | 3.8 | 3.2 | **29.6** |

**观察**：

- Sub-only RL 在 **find / biology / power-component** 上最强，与满分 episode 分布一致。
- **measurement 类**对所有方法都难；Joint MGRPO 均分略高但 valid 极低（14.8%），属于「低质存活」。
- **measure-melting-point** 等 task 在 Sub v2 上均分 -40，RL 也未根本解决。

### 7.5 Joint MGRPO 为何无效

1. **Main 已接近天花板**，联合 RL 预算浪费在 Main 上；组内 Main rollout 常完全相同 → 零 advantage。
2. **kl 极小**（~0.001–0.002），策略更新幅度有限。
3. **未针对 Sub 瓶颈设计**；action valid 在 joint RL 后反而下降（23% → 15% stratified）。

### 7.6 Sub-only RL 为何有效

1. **冻结 Main**，RL 预算全部用于 Sub（~393 万参数）。
2. **更高 action_validity reward**（0.3）直接优化瓶颈指标。
3. **更高 rollout temperature**（0.9）增加组内 contrast，减少零 spread 组。
4. iter_7 在 dev-20 上 valid 38.9%，stratified 上 **50.3%** → 环境动作合法性显著改善。

### 7.7 Sub-only RL 训练动态

- iter 1–7：batch 均分从 -12.6 升至 **54.2**（峰值）。
- iter 8–12：batch 回落并震荡（近 5 iter 均值约 -4.1）。
- kl 始终 < 0.002，**非策略崩溃**，而是 RL 在随机 task batch 上的过拟合/高方差。
- **停训合理**：iter_7 是最佳 checkpoint，继续训练无明确收益。

### 7.8 格式解析

所有 checkpoint 的环境评测 **format_error_rate = 0%**——格式层已解决，问题集中在 **action 语义与环境合法性**。

---

## 8. 实现问题与修复

早期 Joint MGRPO 出现 **loss ≈ 0、advantage 全零**，根因与修复：

| # | 问题 | 修复 |
|---|------|------|
| 1 | 每组只采 1 条 rollout（应为 4 条同 task+var） | `sample_iter_specs` 正确重复 group_size |
| 2 | Rollout 贪心解码 → 组内轨迹完全相同 | `--rollout-do-sample` 默认 True |
| 3 | `group_key` 缺少 `variation_id` | `rollout_schema.py`：`dev:task:var_id` |
| 4 | Sub 使用 rollout 级 advantage 而非 slot 级 | `mgrpo_batch.py` 独立 Sub normalization |
| 5 | Sub 续训无法从 adapter 初始化 | `sft_trainer.py` 新增 `--init-adapter` |
| 6 | Sub-only RL 需冻结 Main | `configure_adapter_training()` |

修复后训练指标正常：`adv_nonzero=100%`，组内 `adv_std≈1`，loss 非零。

---

## 9. 结论与建议

### 9.1 总体结论

| 问题 | 结论 |
|------|------|
| SFT 数据是否 corrupt？ | **否**；问题是数据形态与 hierarchical 部署不匹配 |
| 谁是瓶颈？ | **Sub 执行器**（环境 action valid） |
| Joint MGRPO 是否值得继续？ | **否**；stratified 上 valid 仅 14.8% |
| Sub 续训 SFT 是否有用？ | **是**；dev-20 上 +12pp valid，但是 RL 的好初始化 |
| Sub-only RL 是否有效？ | **是**；stratified 上全面最优，为当前最佳方案 |
| 是否达到可用？ | **否**；145 ep 成功率 9%，测量类 task 大量失败 |

### 9.2 推荐配置

```text
Main: artifacts/checkpoints/sft/main_agent/best
Sub:  artifacts/checkpoints/mgrpo_sub_only/iter_0007/sub
```

### 9.3 下一步建议

1. **评测规范**：后续实验统一使用 **dev stratified-145**（或扩展至 test split）；dev-20 仅作快速 smoke。
2. **数据**：收集环境 rollout 中的 (obs, valid/invalid action) 做 DPO / 负样本 SFT，覆盖 Main 带偏状态。
3. **Reward / 课程**：对 measurement 类 task 加权或单独 curriculum；避免 RL 只优化 find/biology 类。
4. **Sub-only RL 续训**：若再跑 RL，从 iter_7 **resume**，加 early stopping（监控 stratified valid，而非 batch 均分）。
5. **全 test 评测**：正式对外报告前跑完全 test split（~1819 ep）。

---

## 10. 产物索引与复现命令

### 10.1 Checkpoints

| 路径 | 说明 |
|------|------|
| `artifacts/checkpoints/sft/main_agent/best` | SFT Main |
| `artifacts/checkpoints/sft/sub_agent/best` | SFT Sub |
| `artifacts/checkpoints/sft_sub_v2/sub_agent/best` | Sub 续训 1 epoch |
| `artifacts/checkpoints/mgrpo_v2/iter_0007` | Joint MGRPO 最佳 |
| `artifacts/checkpoints/mgrpo_v2/iter_0013` | Joint MGRPO 最终 |
| `artifacts/checkpoints/mgrpo_sub_only/iter_0007` | **Sub-only RL 最佳（推荐）** |
| `artifacts/checkpoints/mgrpo_sub_only/iter_0012` | Sub-only RL 最终（已退化） |

### 10.2 评测结果

| 文件 | 内容 |
|------|------|
| `artifacts/eval/sft_dev20.json` | SFT dev-20 |
| `artifacts/eval/sft_sub_v2_dev20.json` | Sub v2 dev-20 |
| `artifacts/eval/mgrpo_v2_iter07_dev20.json` | Joint MGRPO iter_7 dev-20 |
| `artifacts/eval/mgrpo_v2_iter13_dev20.json` | Joint MGRPO iter_13 dev-20 |
| `artifacts/eval/mgrpo_sub_only_iter07_dev20.json` | Sub-only RL iter_7 dev-20 |
| `artifacts/eval/sft_sub_v2_stratified145.json` | Sub v2 stratified-145 |
| `artifacts/eval/mgrpo_v2_iter07_stratified145.json` | Joint MGRPO stratified-145 |
| `artifacts/eval/mgrpo_sub_only_iter07_stratified145.json` | **Sub-only RL stratified-145** |
| `artifacts/eval/dev_stratified_k5_seed123.json` | 固定 episode list |
| `artifacts/eval/sft_*_offline500.json` | 离线 n=500 |

### 10.3 日志

| 文件 | 内容 |
|------|------|
| `artifacts/mgrpo_v2.log` | Joint MGRPO 训练 |
| `artifacts/mgrpo_sub_only.log` | Sub-only RL 训练 |
| `artifacts/sft_sub_v2.log` | Sub 续训 |

### 10.4 复现命令

```bash
# 生成分层评测列表（145 ep）
bash scripts/generate_stratified_eval.sh

# 正式 stratified 评测
SUB_ADAPTER=artifacts/checkpoints/mgrpo_sub_only/iter_0007/sub \
OUTPUT=artifacts/eval/mgrpo_sub_only_iter07_stratified145.json \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_stratified_eval.sh

# 快速 dev-20 评测
python evaluate_environment.py \
  --base-model Qwen/Qwen3.5-9B \
  --main-adapter artifacts/checkpoints/sft/main_agent/best \
  --sub-adapter artifacts/checkpoints/sft_sub_v2/sub_agent/best \
  --split dev --episodes 20 --no-use-4bit \
  --output artifacts/eval/sft_sub_v2_dev20.json

# Sub-only MGRPO 训练
bash scripts/run_sub_only_mgrpo.sh
```

---

*本报告涵盖 2026-06-12 ~ 2026-06-15 全部实验线、dev-20 与 stratified-145 评测结果及数据分析。*
