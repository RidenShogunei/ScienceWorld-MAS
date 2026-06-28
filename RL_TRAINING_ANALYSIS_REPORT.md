# MRPO 训练失败分析报告

**项目**: ScienceWorld-MAS / MRPO Expert Subtask Contract V3  
**训练时间**: 2026-06-27 ~ 2026-06-28 (约22小时)  
**训练轮次**: iter 01 - iter 06 (共20轮, 已停止于iter 06)  
**Base Model**: Qwen/Qwen3.5-9B  
**评估集**: dev_stratified_k5_seed123 (30 tasks × 5 variations = 145 episodes)

---

## 1. 执行摘要

**结论: 训练已停止，iter 05 为当前最佳版本 (mean=14.16, eval avg=8.0)**

| 指标 | iter 01 | iter 02 | iter 03 | iter 04 | iter 05 | iter 06 |
|------|---------|---------|---------|---------|---------|---------|
| **Rollout Mean** | 0.19 | 6.73 | 7.72 | 5.30 | **14.16** | 2.56 |
| **Rollout Median** | 7.00 | 3.00 | 5.00 | 10.00 | **13.00** | 5.00 |
| **正分率** | 71.9% | 71.9% | 75.0% | 73.4% | **81.2%** | 71.9% |
| **-100分率** | 18.8% | 7.8% | 6.2% | 21.9% | 6.2% | **12.5%** |
| **训练Loss** | - | - | - | - | +0.0006 | **-0.0353** |

**关键发现**:
1. iter 05 → iter 06 出现 **loss=-0.0353 (负数)**，策略更新方向错误
2. 57-71% 的 sub-agent 动作被标记为 `action_valid=False`（环境不认可的动作）
3. 奖励Landscape是"悬崖型" — 选对物体得 +50，选错得 -100~-183
4. 每轮只采样 8/30 个任务，任务覆盖严重不足
5. **训练无明显上升趋势**，iter 05 的"高峰"可能是随机因素

---

## 2. 根因分析

### 2.1 奖励函数设计问题：悬崖型奖励 Landscape

ScienceWorld 的奖励机制对"正确物体选择"极度敏感：

```
任务: lifespan-longest-lived-then-shortest-lived

正确动作:
  focus on egg giant tortoise  → +50
  focus on egg parrot         → +50  
  focus on baby ant           → +17  (最短命的动物)

错误动作（惩罚极端）:
  focus on baby hedgehog      → -88.7
  focus on baby beaver        → -126.4
  focus on juvenile beaver     → -133
  focus on juvenile giant tortoise → -183
```

**问题**: 模型无法从分数反馈中学习"为什么错"。同样都是"focus on X"，但 X 的细微差异导致 +50 和 -183 的天壤之别。这种稀疏、悬崖型的奖励使得梯度信号极不稳定。

### 2.2 Sub-Agent 产生幻觉动作

**57-71% 的动作被标记为 `action_valid=False`**（环境不认可该动作）：

```
iter 1: 57.2% invalid
iter 2: 64.0% invalid
iter 3: 70.7% invalid
iter 4: 69.3% invalid
iter 5: 66.3% invalid
iter 6: 70.7% invalid
```

典型幻觉动作案例：
```
"focus on baby baby baby baby baby baby baby baby baby baby baby baby baby baby baby cat"
"pick up glowing blue potion"
"pick up infrared thermometer"
"open door to baby"
```

这些动作根本不存在于环境中，说明 **sub-agent 没有理解物体命名约束或空间推理能力不足**。

### 2.3 任务随机采样导致覆盖不足

每轮训练只采样 **8 groups × 8 rollouts = 64 samples**，从 30 个任务中随机选 8 个：

| 任务 | 出现轮次 |
|------|---------|
| lifespan-longest-lived-then-shortest-lived | iter 1, 4, 6 (稀疏) |
| power-component | iter 1, 4, 5 |
| measure-melting-point-known-substance | iter 2, 3, 5 |
| boil | iter 2, 6 |

**结果**: 大部分任务每 3-6 轮才被采样一次，模型没有足够的迭代学习每个任务。

### 2.4 Group-Based Advantage Normalization 的问题

每组 8 个 rollouts 共享 (task, variation)，然后做 advantages normalization：

```
Group dev:lifespan-longest-lived-then-shortest-lived:83:
  8 rollouts: [-100, -100, -100, 33, 83, 83, 100, 100]
  adv_std=1.0000
  adv=[-1.153, -1.153, -1.153, 0.5, 0.9, 0.9, 1.2, 1.2]
```

当 -100 分的 rollout 出现时，它会把整组 advantages 拉偏，导致：
- 正确得了 33 分的 rollout advantage 被压低
- 错误的 -100 分 rollout advantage 不会足够负（因为被 normalize 了）

### 2.5 为什么 iter 05 → iter 06 暴跌？

**核心问题是 iter 06 采样到了更差的任务组合**：

| iter 05 采样的任务 | mean | iter 06 采样的任务 | mean |
|-------------------|------|-------------------|------|
| power-component | 45.4 | chemistry-mix-paint-secondary-color | 22.5 |
| find-plant | 19.0 | find-living-thing | 21.0 |
| mendelian-genetics-known-plant | 18.6 | lifespan-longest-lived-then-shortest-lived | 12.4 |
| measure-melting-point-known-substance | -27.4 | **boil** | **-11.0** |
| | | **mendelian-genetics-unknown-plant** | **-10.2** |

iter 06 采样的任务整体更差，加上：
- `-100` 分率从 6.2% 翻倍到 12.5%
- 训练 loss 变成 **负数 (-0.0353)**，策略往错误方向更新

---

## 3. 具体案例分析

### 案例 1: `lifespan-longest-lived-then-shortest-lived` (var=83) — 同一任务的8次完全不同命运

**iter 06 中该任务有8个rollouts，结果天差地别**:

| Rollout | Final Score | 关键动作 |
|---------|-------------|---------|
| r0000 | **-100** | focus on juvenile giant tortoise → -183 |
| r0004 | 83 | focus on giant tortoise → +50 |
| r0009 | **100** | focus on egg giant tortoise → +50, focus on baby ant → +17 |
| r0013 | **-100** | focus on giant tortoise (again) → -183 |
| r0034 | 83 | focus on egg giant tortoise → +50 |
| r0038 | 33 | 混乱的导航，无明确策略 |
| r0042 | **100** | focus on giant tortoise → +50, focus on juvenile ant → +17 |
| r0048 | **-100** | focus on giant tortoise → -183 |

**分析**:
- 同一个 variation (83)，模型在 iter 06 中时而得 +50、时而得 -183
- 关键区别: `focus on giant tortoise` vs `focus on egg giant tortoise` vs `focus on juvenile giant tortoise`
- 同样是 "focus on X"，但 X 的 life stage 决定 +50 还是 -183
- 模型无法可靠地学习"何时该用 egg/juvenile/adult"

### 案例 2: `measure-melting-point-known-substance` — -140 的诱惑

**iter 05 中该任务4次出现 -100**:

```
Rollout r0035 (final=-100):
  sub 0: go to hallway → +1
  sub 2: go to kitchen → +5, pick up thermometer → +1
  sub 6: focus on thermometer in inventory → +33
  sub 8: focus on thermometer in inventory → **-140**

Rollout r0041 (final=-100):
  sub 0: go to hallway → +1  
  sub 2: go to kitchen → +5, pick up thermometer → +1, focus thermometer → +33
  sub 4: focus on thermometer → **-140**

Rollout r0051 (final=-100):
  sub 3: focus on inventory → **-100** (直接瞄准inventory!)
```

**分析**:
- 同一个 `focus on thermometer in inventory` 动作有时 +33 有时 -140
- `focus on inventory` (没有指定物体) 直接 -100
- 模型在**探索**过程中被极端负奖励惩罚，然后这种策略被 disadvantage

### 案例 3: `boil` — 任务理解彻底失败

**iter 06 中 boil 任务的8个rollouts**:

| Rollout | Final Score | 得分动作 |
|---------|-------------|---------|
| r0005 | 2 | activate stove → +2 |
| r0031 | 0 | (无任何奖励) |
| r0044 | 2 | activate stove → +2 |
| r0046 | **-100** | activate stove → +2, then focus on thermometer → -102 |
| r0053 | 2 | activate stove → +2 |
| r0054 | 2 | activate stove → +2 |
| r0055 | 2 | activate blast furnace → +2 |
| r0057 | 2 | activate stove → +2 |

**任务描述**: "Your task is to boil lead."

**分析**:
- 所有 rollout 都卡在 `activate stove → +2`，没有推进到真正的 boiling lead 步骤
- 唯一得了 -100 的是误触了 `focus on thermometer`，触发了 -102 的惩罚
- **模型不理解"boil lead"需要先 pick up lead，然后放到 stove 上加热**

### 案例 4: Action Validity — 100% 动作都是幻觉

**iter 06 rollout r0061** (`power-component-renewable-vs-nonrenewable-energy`):
```
50个动作，全部被标记为 action_valid=False

'open door to kitchen', 'go to hallway', 
'pick up generator', 'pick up generator',  // 重复
'pick up glowing blue potion',  // 不存在
'pick up metal pot',
'pick up metal pot',
```

**分析**:
- 模型在"抓取"这个动作上极度混乱，产生大量幻觉物体名
- `format_valid` 接近 100%（模型知道输出格式）
- 但 `action_valid` 极低（模型不知道哪些物体实际存在）

---

## 4. 训练动态追踪

### 4.1 训练曲线

```
iter   mean    median    std      pos%    -100%
  1     0.19     7.0    54.0     71.9%    18.8%
  2     6.73     3.0    37.7     71.9%     7.8%
  3     7.72     5.0    35.8     75.0%     6.2%
  4     5.30    10.0    63.3     73.4%    21.9%  ← 采样到高难度任务
  5    14.16    13.0    38.2     81.2%     6.2%  ← 峰值
  6     2.56     5.0    44.7     71.9%    12.5%  ← 崩溃
```

**观察**:
- **无明显上升趋势**，iter 05 的峰值更像是随机因素（采样到了 power-component 这样的简单任务）
- **高方差** 持续存在（std=35-63），训练极不稳定
- **-100分率** 在 6-22% 徘徊，无法收敛

### 4.2 训练 Loss 异常

| Iteration | Main Loss | KL | Clip Rate |
|-----------|-----------|-----|-----------|
| iter 05 | +0.0006 | 0.1412 | 22.43% |
| iter 06 | **-0.0353** | 0.1374 | 21.47% |

**iter 06 出现负 loss**，这意味着：
- 策略在增加坏动作的概率
- Advantage 估计可能出了问题
- 可能因为 -100 分样本的 advantage normalization 失真

---

## 5. 建议改进方向

### 5.1 奖励重塑 (Reward Shaping)

当前: 只有最终正确/错误动作给分
建议: 加入中间奖励

```
Level 1: 导航到正确房间      → +5
Level 2: 拾取正确物体        → +10  
Level 3: 执行正确动作        → +15
Level 4: 正确物体 + 正确目标 → +50 (最终)
Level 5: 错误物体            → -5 (轻微惩罚，防止乱猜)
Level 6: 严重错误 (focus on inventory) → -20
```

### 5.2 降低任务难度

当前: 30 个任务，全部一起训
建议: 分课程学习

```
Phase 1 (5个简单任务): navigation, open door, pick up basic objects
Phase 2 (10个中级任务): 简单科学推理 (boil water, freeze, melt)
Phase 3 (全部30个任务): 复杂多步推理
```

### 5.3 减少 Group Size

当前: 8 groups × 8 = 64 rollouts per iteration
建议: 4 groups × 16 rollouts = 64 rollouts

单组内 8 个样本过多，当出现单个 -100 时会严重扭曲 advantage normalization。

### 5.4 改进 Advantage 估计

当前: Group-based GAE
建议: 
- 使用 TD(λ) 或gae_with_whiten
- 对 -100 分样本做截断或特殊处理
- 降低极端值对 normalization 的影响

### 5.5 改进 Sub-Agent Prompt

当前: 简单的 contract 格式
建议:
- 明确限制可操作的物体列表
- 加入 few-shot examples 说明正确动作格式
- 对 hallucinated 动作加强惩罚

### 5.6 增加任务覆盖

当前: 每轮只随机选 8/30 个任务
建议:
- 改成 curriculum sampling，优先补足未充分训练的任务
- 或增加每轮采样数到 16 groups

---

## 6. 结论

**MRPO 在 ScienceWorld 上的失败不是某个单一原因造成的，而是多个系统性问题叠加的结果：**

1. **奖励悬崖** — 正确/错误动作之间没有平滑过渡，梯度信号极不稳定
2. **动作幻觉** — 70% 的 sub-agent 动作是环境不认可的，模型在错误空间探索
3. **任务欠覆盖** — 每轮只训 8/30 的任务，已训任务会遗忘
4. **Advantage 失真** — -100 分对组内 normalization 的影响不成比例
5. **无课程学习** — 简单任务和困难任务混在一起，互相干扰

**iter 05 虽然是当前最佳 (rollout mean=14.16, eval avg=8.0)，但这个"最佳"更像是随机因素而非训练收敛**。如果继续训练到 iter 20，大概率会继续震荡或退化。

**建议**: 在改进奖励设计和任务课程之前，不建议继续当前训练配置。

---

*报告生成时间: 2026-06-28*
*数据来源: /home/jinxu/ScienceWorld-MAS/artifacts/checkpoints/mgrpo_expert_subtask_contract_v3_mrlx_like_v1/*
