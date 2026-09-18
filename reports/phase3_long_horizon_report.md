# 第二阶段报告 —— 正常肌力长时行走稳定性：诊断、微调入口与有限预算试训练

> 基线：`196b748`。本报告所有数字均由**实际执行**产生，来源可查。
> 生成时间：2026-09-18。验证环境：conda `hemirl`（Python 3.12.13, mujoco 3.11.0,
> numpy 2.3.5, gymnasium 1.2.3, **torch 2.14.0+cpu**, SB3 2.7.1）。
> 本次**固定所有肌力倍率为 1.0**，不加拐杖、不做偏瘫损伤训练、不启动临界点扫描。

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 长时失稳诊断 | ✅ 完成。给出可解释的因果链与量化证据（`reports/instability_diag.json`） |
| `group_value_repeat_ok=False` 核查 | ✅ 确认是**公式错误**，非策略加载问题；已修正并纳入通过条件（现 8/8） |
| 训练/评估/续训入口 | ✅ 已实现（`scripts/train_healthy.py`、`scripts/eval_healthy.py`、`configs/train_healthy_v1.json`） |
| 配置与资源开销 | ✅ 已实测（`reports/throughput.json`、`reports/train_check_init.json`） |
| 训练路径检查（2000 transitions × 4 组初始化） | ✅ 通过；**重置 critic 会立刻摧毁行走**，故沿用官方 critic/alpha |
| 有限预算试训练 | ✅ 完成，**未超时**（12.51 分钟 / 30 分钟），transition 数超出上限 604 步（已如实记录） |
| 配对评估（A 官方 vs B 微调，20 测试种子） | ✅ B **全面改善**：存活 +0.62 s、速度 +0.072 m/s、侧偏 −0.156 m、倾角 −31.7° |
| **是否达到进入损伤适应训练的条件** | ❌ **未达到**。门槛要求 18/20 完成 20 s 行走，实测 A=0/20、B=0/20 |
| 下一次单项调整 | 给出 1 项（含可执行命令），见 §8 |

---

## 1. 长时失稳诊断

### 1.1 诊断口径（哪些量、怎么看）

`hemirl/instability_diag.py` 逐步记录：骨盆世界侧向位置/速度/倾角/角速度、全身质心
（`data.subtree_com[0]`）位置与差分速度、足部接触与滑动（**只有法向力 > 20 N 才算支撑**，
纯 geom 接触会把脚尖轻擦算进来，实测假滑动可达 2.4 m/s）、左右下肢激活与动作饱和、
各奖励分量、以及**分组**的原始/标准化观测与裁剪情况。

**参考自身 vs 人体漂移必须分开**（任务书要求）：

* `ref_*`：参考自己的位置波动 → 反映上游轨迹的固有属性；
* `drift_*`：人体相对参考的偏差 → 反映人体是否真的在积累误差。

### 1.2 观测裁剪：先把判据写对

`VecNormalize.normalize_obs` **内部直接** `np.clip` 到 ±`clip_obs`，因此看它的输出
永远得到 `|z| ≤ 10`，「裁剪比例」恒为 0。必须自己算

$$z_{\text{raw}} = \frac{obs - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

再与 `clip_obs` 比较。按修正后的判据，各观测分组（5 seed 的逐 seed 最大值）：

| 观测分组 | dim | 最大被裁剪维数 | 裁剪比例 | `z_raw` 最大 |
|---|---|---|---|---|
| `qpos` | 85 | 11 | 12.9% | **33.2** |
| `qvel` | 85 | 16 | 18.8% | 18.2 |
| `qacc` | 85 | 14 | 16.5% | 102.9 |
| `act` | 700 | 69 | 9.9% | 107.2 |
| `actuator_forces` | 700 | 72 | 10.3% | 517.3 |
| `actuator_length` | 700 | 77 | 11.0% | 45.7 |
| `actuator_velocity` | 700 | 55 | 7.9% | 47.9 |
| `key_xpos` | 18 | 7 | **38.9%** | 68.6 |
| **`qpos_ref`** | 85 | **0** | **0%** | 3.20 |
| **`qpos_ref_future`** | 425 | **0** | **0%** | 3.21 |
| **`key_xpos_ref`** | 18 | **0** | **0%** | 2.21 |

**参考类分组从不被裁剪；状态类分组被裁剪。** 重点关注分量的 `z_raw`：

| 分量 | `z_raw` 最大 |
|---|---|
| `root_abs_pos`（`qpos[0:3]`，绝对根位置） | 9.5 – 17.8 |
| `root_abs_rot`（`qpos[3:6]`） | 14.6 – 21.7 |
| `root_abs_vel`（`qvel[0:3]`） | 5.5 – 7.7 |
| `ref_cur_pos` / `ref_next_pos` | 3.1 / 3.1 |

绝对根位置的越界主要由**侧向分量**驱动：参考的侧向位置只摆动 ±0.05 m，因此该维在
训练统计里的 σ 极小；人体一旦侧向偏出 0.4–0.5 m，`z_raw` 立刻到 10–18。

**异常是否出现在失稳之前？——是，但要分清是哪一组：**

* `act` / `actuator_forces` / `qacc` 从 t = 0.08–0.72 s 就已在裁剪 → 属**正常工况**；
* `qpos` / `qvel` / `key_xpos` / `actuator_length` / `actuator_velocity` 的**首次裁剪都在
  4.04–4.12 s** → 与跌倒同刻，是结果。

### 1.3 参考自身 vs 人体实际漂移

| 量（世界 y，+y 为侧向） | 值 |
|---|---|
| 参考自身 `ref_pelvis_y` \|max\| | **0.076 m** |
| 参考自身相对起始 `ref_y_minus_ref_y0` \|max\| | 0.046 m |
| 人体 `pelvis_y` \|max\| | **0.514 m** |
| 人体相对参考 `drift_y_human_minus_ref` \|max\| | **0.438 m** |
| 人体相对参考前向 `drift_x` 末值 | −0.63 m |

人体侧向漂移是参考自身摆动的 **5.8 倍** → 侧向失稳是**人体行为**，不是参考在漂。

**参考循环边界**（`reports/reference_continuity.json`）：

* 周期内单控制步参考变化最大 0.0848，循环边界处 0.0677 →
  **边界跳变比正常单步更小**，没有异常跳变；
* 但轨迹**非严格周期**：首末帧姿态差 0.0329 rad；且因
  `clip(time_step, 0, num_frames-1)` 帧序列只覆盖 1.16 s 而 `terminate_time = 1.17 s`，
  **每周少一帧（0.01 s）**；前进位移累积正确（相对误差 4.0e-4）。

> ⚠️ **不因此宣布「完全排除参考因素」**。边界小只能说明「没有大跳变」，不能排除
> 参考的其他属性。§5 的 E3 显示：把参考超前一个控制步去掉，动作平均改变 `0.056`、
> 最大 `1.41` —— 这是个**非平凡**的敏感度。参考因素作为**候选**保留，见 §7。

### 1.4 领先指标排序（谁先动）

对每个信号用回合前 1.0 s 建立基线（mean ± 6σ），找首次越界（5 seed 中位）：

```
action_sat_frac            1.42 s
action_sat_frac_L_lower    1.78 s
foot_slip_max              2.54 s
drift_y_human_minus_ref    3.58 s
qpos_track_err             3.98 s
pelvis_w_norm              4.06 s
up_tilt_deg                4.09 s   ← 跌倒判据触发
```

注意 `action_sat_frac` 在**前 1 s 均值已达 0.077**（7.7% 的动作维钉在 |a| ≥ 0.99），
末段升到 ~1.0。所以「1.42 s 首次越界」指的是相对已饱和基线的**变化**，不是饱和的开始。

### 1.5 奖励分量（解释「为什么只靠存活项发现不了」）

| 分量 | 前 1 s | 2–3 s | 末段 |
|---|---|---|---|
| `imitation` | −15.6 | −16.6 | **−788.8** |
| `energy` | −3.1 | −3.0 | −11.1 |
| `official_healthy` | +100 | +100 | **0** |
| `survival_physical` | +100 | +100 | **+100** |

`survival_physical` 直到最后一刻都是 +100（骨盆高度直到跌倒也 > 0.55 m），
而 `imitation` 早已从 −15.6 恶化到 −789。→ **只靠「物理存活」项无法在早期发现侧向失稳**。

---

## 2. `group_value_repeat_ok` 字段核查（任务书指定）

**结论：该字段是公式错误，与「策略加载有误」无关。**

* 旧公式：`np.array_equal(a, a[group_of_muscle])`，即要求 `g[g[i]] == g[i]`
  （`g` 为 `group_of_muscle`），也就是要求「肌肉下标恰好等于组号」这一**巧合**成立；
* 实测：真实 checkpoint 的 **138 个组组内全部同值**（混合值的组数 = 0），
  而旧公式报 **697/700** 个元素「不等」；
* 该字段**原本不在** `checks` 字典里，所以 `passed=True` 与
  `group_value_repeat_ok=False` 并不矛盾——但字段名与实际检验内容不符。

已修正为正确判据 `a == a[first_of_group[group_of_muscle]]`（组内同值），
并把 `group_value_repeat_ok` **纳入通过条件**。修正后 `reports/verify_policy.json`：

```
[PASS] group_value_repeat_ok     （n_groups_with_mixed_values = 0，旧公式 697 个假阳性）
8/8 检查通过，max|Δa_sb3−a_direct| = 5.960e-08 ≤ 1e-5（126 个样本）
```

---

## 3. 训练 / 评估 / 续训入口

### 3.1 入口清单

| 用途 | 入口 |
|---|---|
| 正常行走微调 | `scripts/train_healthy.py --config configs/train_healthy_v1.json` |
| 续训 | 同一命令加 `--resume` |
| 训练路径检查 | `--check-only --check-transitions 2000` |
| 配对评估 | `scripts/eval_healthy.py --policies A=<官方> B=<微调>` |
| 训练前后对比 | `scripts/compare_policies.py --policies A=<官方> B=<微调>` |
| 长时失稳诊断 | `scripts/diagnose_instability.py` |
| 命名对照实验 | `scripts/named_controls.py` |
| 资源与吞吐实测 | `scripts/measure_throughput.py` |

训练与评估**都**通过 `hemirl.research_env.build_research_env()` 构建，
终止/计时/奖励语义一致；沿用上游 `DynSyn.SAC_DynSyn.SAC_DynSyn` 与 `Actor_DynSyn`，
不引入新算法、不改观测（3601）与动作（700）结构。

### 3.2 各项参数的加载 / 初始化方式（全部落盘在 `train_meta.json`）

| 对象 | 处理 |
|---|---|
| **actor** | 从官方 checkpoint 加载（含 `dynsyn_layer` 的 138 维输出头） |
| **critic** | `critic_init=keep` → 沿用官方；`reset` → 只重置 `critic.q_networks` 的 Linear 层 |
| **critic_target** | 与 critic 同步（`load_state_dict`） |
| **温度 α** | `alpha_init=keep` → 沿用（实测 0.1975）；`reset` → `log_ent_coef=0` 且重建优化器 |
| **actor/critic 优化器** | **重建 Adam**，显式 lr=1e-4；不继承 checkpoint 的优化器状态 |
| **ent_coef 优化器** | keep 时沿用 / reset 时重建 |
| **lr 与进度调度** | **显式替换**为常数 1e-4；不继承已完成训练的旧进度（继承值本为 1.25e-4 @ progress 0.1） |
| **`dynsyn_weight_amp`** | **显式固定 0.0**（见 3.3） |
| **VecNormalize** | 加载官方统计，训练与评估都 **冻结**（`training=False`） |
| buffer / 节奏 | `buffer_size=1e5`、`batch_size=256`、`train_freq=2`、`gradient_steps=1`、`learning_starts=200` |

**critic 重置的两个坑**（都已规避）：
(a) SB3 的 `ContinuousCritic` **没有** `reset_parameters()`；
(b) 不能对整个 critic 递归重置 —— `features_extractor` 与 actor **共享**，递归会把 actor 一起重置。

### 3.3 `dynsyn_weight_amp` 的静默语义切换（必须显式固定）

checkpoint 里 `dynsyn_weight_amp = None`，而 `dynsyn_k = 5e-9`、`dynsyn_a = 3e7`、
`num_timesteps = 4.5e7`。上游 `SAC_DynSyn.train()` 每轮会执行

```python
dynsyn_weight_amp = self.get_dynsyn_weight_amp(self.dynsyn_k, self.dynsyn_a, self.num_timesteps)
# = min(max(0, 5e-9 * (4.5e7 - 3e7)), 0.1) = 0.075
self.actor.dynsyn_layer.update_dynsyn_weight_amp(dynsyn_weight_amp)
# DynSynLayer.forward: weight = clamp(weight*0.1, -amp, amp) + 1
```

即 **amp = 0.075**，肌群协同比会被重标定 —— **动作语义被静默改变**。
本项目固定 `dynsyn_weight_amp = 0.0`，此时 `weight ≡ 1`，与官方基线完全一致
（该等价关系已由 `reports/verify_policy.json` 的 amp 语义检查证实：`amp=None` ≡ `amp=0`）。
实测训练前后 `layer.dynsyn_weight_amp` 恒为 `0.0`。

### 3.4 续训与产物保留

* 保存按官方 checkpoint 目录布局（`checkpoint/best_model.zip` + `best_env.zip` +
  `locomotionFull.json`），因此 `LocomotionEvaluator` / `eval_official.py` 可直接评估微调产物；
* 同时保存 `last_*` 供 `--resume`；保存/重载一致性实测 `True`（动作逐元素 < 1e-6）；
* 每次运行追加到 `runs/train/<tag>/train_history.json`（append-only），
  因为续训会覆盖 `train_meta.json`。
* 大体积产物已排除出版本控制：单个 zip 约 275–288 MB（`policy.pth` 130 MB +
  `critic.optimizer.pth` 104 MB + `actor.optimizer.pth` 53 MB，**不含 replay buffer**），
  `.gitignore` 增加 `runs/**/checkpoint/`。

---

## 4. 配置与资源开销（实测）

`reports/throughput.json`（20 核 CPU、31 GiB RAM、**无可用 GPU**：
`nvidia-smi` 报 NVML driver/library mismatch，且 torch 为 CPU 版）

| 项 | 实测 |
|---|---|
| 环境推进（单进程） | 298 steps/s |
| 环境推进（4 / 8 / 12 并行） | 1150 / 1942 / **2457** steps/s（次线性） |
| **完整 SAC 更新** | **3.6 次/s（275 ms/次）**；batch 128 与 256 几乎同速 → 瓶颈是固定开销/内存带宽 |
| replay buffer | **31,613 B/transition**（obs 3601 + next_obs 3601 + act 700，float32） |
| buffer 容量 | 1e4 = 0.29 GiB / 1e5 = **2.94 GiB** / 2e5 = 5.89 GiB / **1e6 = 29.44 GiB** |

**两处不能沿用上游默认值的地方**：

1. **buffer**：上游 `buffer_size = 1e6` 需要 29.44 GiB ≈ 本机全部内存 → 改为 `1e5`（2.94 GiB）。
2. **并行数**：上游 `n_envs = 64` 在本机（20 核）会严重超订 → 实测最优取 **12**。

**一个被纠正的公式**：SB3 的 `OffPolicyAlgorithm.learn` 是「每轮 `collect_rollouts` 之后
调用一次 `train()`」，而一轮收集 `train_freq × n_envs` 个 transition，因此

$$n_{\text{updates}} = \frac{\text{transitions}}{\text{train\_freq} \times n_{\text{envs}}}$$

**不是** `transitions / train_freq`。最初按后者估算，会高估更新次数 `n_envs` 倍
（用 `train_freq=16` 时实际只有 260 次更新，策略几乎不动）。最终取 `train_freq=2`。

**另一个观察**：训练时主进程 torch 默认吃满约 14.5 核，12 个环境 worker 各只剩约 11% 核。
在 `train_freq=2` 下梯度更新仍是瓶颈（2.81 次/s），因此未再调整线程数。

---

## 5. 训练路径检查与命名对照实验

### 5.1 2000-transition 检查（含 critic/α 初始化对照）

`reports/train_check_init.json`。协议：2000 transitions、12 env、`train_freq=2`、
batch 256；指标为验证集（seed 101–105，6 s 回合）平均存活与固定 40 个探针观测上的策略漂移。

| critic | alpha | 策略漂移 mean\|Δa\| | 验证集平均存活 |
|---|---|---|---|
| **keep** | **keep** | **0.156** | **4.95 s** |
| keep | reset | 0.267 | 4.98 s |
| reset | keep | 0.443 | **0.64 s** ← 崩溃 |
| reset | reset | 0.562 | **0.70 s** ← 崩溃 |

**结论（有对照，不是假定）**：**重置 critic 会立刻摧毁已学会的行走**。
理由与「奖励量级」一致：split 奖励（`imitation + energy + survival_physical`）
实测约 **+81.3/步**，与官方奖励同量级，价值函数无需重建。故采用 `keep + keep`。

**同时通过的其他检查项**：actor/critic 梯度数值有限 ✅；终止/截断/timeout 由
`research_env` 统一处理 ✅；checkpoint 可保存并重载且动作逐元素一致 ✅；
归一化统计随模型保存 ✅；评估环境不更新统计 ✅（`training=False`，
`compare_policies` 实测两个 checkpoint 的 `obs_rms` **逐位相同**）；
动作语义与配置一致 ✅（`dynsyn_weight_amp` 训练前后恒为 0.0）。

### 5.2 命名对照实验（每次只改一个因素）

`reports/named_controls.json`。

**E1 观测裁剪（`clip_obs` 10 → 1e9），固定策略，配对种子 0–4，20 s：**

| 设置 | 平均存活 | 跌倒 |
|---|---|---|
| `clip_obs = 10` | 4.62 s | 5/5 |
| `clip_obs = 1e9`（不裁剪） | 4.61 s | 5/5 |

→ **取消裁剪没有任何改善**。结合 §1.2 的时序（`qpos` 的首次裁剪在 4.04–4.12 s，
而侧向漂移从 ~2.5 s 就开始），结论：**裁剪是侧向漂移的结果，不是原因**。

**E2 裁剪对动作的影响（纯前向，5 seed）：** `mean|Δa| = 0.0003–0.0017`（可忽略），
但 `max|Δa| ≈ 2.0`（个别维完全翻转）；平均 **~19 个**观测维被裁剪。
→ 影响集中在少数维上，整体动作变化很小，与 E1 的「无改善」一致。

**E3 参考超前一个控制步（纯前向，5 seed）：** `mean|Δa| = 0.056`，
`max|Δa| = 1.08–1.41`。构造方式为精确单因素：第 k 步的「无超前」参考块
**恰好等于第 k−1 步的参考块**，只替换观测里的参考切片。
→ 参考超前对动作有**非平凡**影响（平均 5.6%），因此**不能排除**参考因素，
见 §7 的候选清单。

---

## 6. 有限预算试训练结果

### 6.1 预算与运行

| 项 | 值 |
|---|---|
| 主试训练 | **50,004 transitions / 12.37 分钟**（67.3 transitions/s），2,083 次梯度更新（2.81 次/s） |
| 续训验证 | 600 transitions / 0.14 分钟 |
| 合计 | **50,604 transitions / 12.51 分钟**（30 分钟预算内） |
| 停止原因 | 到达 transition 预算（两侧均未触发墙钟上限） |
| 课程 | 停留在 **s6（6 s）**，未升级 |
| 数值健全性 | actor/critic 参数全部有限；α: 0.1975 → 0.2343 |
| `dynsyn_weight_amp` | 训练前后恒为 **0.0**（未发生语义切换） |
| 保存/重载一致性 | `True` |

⚠️ **如实记录两处偏离**：

1. **transition 数超出 50,000 上限 604 步（+1.2%）**，原因是需要实际验证 `--resume` 路径。
   当前 tag 下的模型是累计 50,604 transitions 的版本，配对评估用的就是它。
   若要严格不超上限，删除 `runs/train/healthy_v1/` 后不加 `--resume` 重跑主命令即可。
2. **续训覆盖了主试训练的产物**（`last_model.zip`/`best_model.zip`/`train_meta.json`）。
   两次运行的完整记录已补写到 `runs/train/healthy_v1/train_history.json`，
   并且脚本已改进为 append-only 记录，后续不会再丢历史。

### 6.2 课程为什么没有升级

升级条件是「至少攒够 `min_transitions` **且** 独立验证集成功率 ≥ 0.8」。
验证集（seed 101–105，6 s 回合）成功率始终为 **0.00**，平均存活 5.07 s（主试后）
→ 5.51 s（续训后），即**连 6 s 都没走完**，因此正确地在第 1 阶段停下。

---

## 7. 配对评估：A 官方 vs B 微调

### 7.1 协议

20 个**最终测试种子 201–220**（与验证种子 101–105 分离，且未据测试结果调参）、
20 s 回合、确定性推理、配对种子、归一化冻结、**不中途 reset、不覆盖根节点、不加稳定力**。
判据要求「在走」与「没倒」同时成立（完成时长 + 速度 + 侧偏 + 倾角 + 无数值异常）。

### 7.2 结果

| 指标 | A 官方 | B 微调 | 配对差值 |
|---|---|---|---|
| **20 s 完成率** | 0/20 | 0/20 | 0 |
| 平均存活时间 | 4.97 s | **5.60 s** | **+0.62 s** |
| 平均前进速度 | 0.893 m/s | **0.965 m/s** | **+0.072 m/s** |
| 目标速度误差 | 0.145 | **0.073** | −0.072 |
| 最大侧偏（相对参考） | 0.450 m | **0.294 m** | **−0.156 m** |
| 最大骨盆倾角 | 61.1° | **29.4°** | **−31.7°** |
| 足部滑动均值 | 2.57 m/s | 2.74 m/s | +0.17 |
| 肌肉激活均值 | 0.1922 | 0.2099 | +0.018 |
| 动作饱和比例 | 0.1288 | **0.1248** | −0.0040 |
| 退化「站着不走」 | 0 | 0 | 0 |
| 数值异常 | 0 | 0 | 0 |
| 终止来源 | `physical_fall` 20/20 | `physical_fall` 20/20 | — |

**失败模式也变了**：A 有 16/20 因**骨盆直立偏差**触发（倾角 59–63°），
B **20/20 全部因骨盆高度**触发（倾角仅 20–40°）——微调后的策略不再「倾倒」，
而是「下沉」。

### 7.3 训练前后对比（`reports/train_vs_baseline.json`）

| 项 | 结果 |
|---|---|
| 归一化统计 | 两个 checkpoint **逐位相同**（`obs_rms` 最大差 0.0）✅ |
| 策略变化 | `mean|Δa| = 0.352`，`max|Δa| = 2.0`（在含跌倒的轨迹上） |
| 动作饱和 | A 0.1485 → B 0.1953 |
| **价值估计**（同状态-动作对 `(z, a_A)`） | `Q_A(t0)=4227.5` vs 官方奖励 MC 回报 4274.1 → **比值 0.989**（校准良好）<br>`Q_B(t0)=3889.3` vs 新奖励 MC 回报 2859.1 → **比值 1.360**（高估 36%） |
| 奖励量级（同一条 A 轨迹） | 官方合计 **−28.4/步**；微调合计 **−38.7/步** |

**逐分量量级（同一条 A 轨迹均值）**：

| 分量 | 值/步 |
|---|---|
| `imitation` | −105.6 |
| `energy` | −4.0 |
| `official_healthy` | +81.2 |
| `survival_physical` | **+100.0** |
| `extra_lateral_dev` | −9.0 |
| `extra_lateral_vel` | −6.8 |
| `extra_forward_shortfall` | −7.3 |
| `extra_slip` | −6.0 |
| `extra_total` | **−29.1** |

### 7.4 归因（基于上述证据，不把失败改写成成功）

1. **有实质改善，但没有解决问题**：存活 +0.62 s、侧偏 −0.156 m、倾角 −31.7°、
   速度 +0.072 m/s，全部同向改善；然而 20/20 仍跌倒，完成率仍为 0。
2. **价值函数仍在适应**：`Q_B/G_new = 1.360`（高估 36%），而官方 critic
   对官方奖励的比值是 0.989。2,083 次更新（`batch 256`、`buffer 1e5`）远不足以
   让 critic 收敛到新的回报尺度 → 后续训练预算必须显著加大。
3. **奖励结构仍偏向「保高度」而非「守路径」**：`survival_physical` 给 **+100/步**，
   而侧向类惩罚合计仅 **−15.8/步**（`lateral_dev` + `lateral_vel`），
   **相差约 6 倍**。策略于是理性地选择「保持骨盆在 0.55 m 以上」，代价是路径精度——
   这与观测到的「倾角大幅下降、失败模式改为下沉」完全一致。
4. **参考因素不能排除**：E3 显示去掉参考超前会让动作平均改变 5.6%。但 §6.1 的
   微调实验本身是对参考因素最有力的检验——在**完全相同的参考**上，仅优化策略就把
   存活从 4.97 s 提到 5.60 s，说明失败至少部分可由策略优化解决，而不是参考的固定缺陷。

---

## 8. 明确结论与下一次调整

### 8.1 是否达到进入损伤适应训练的条件？

**否。** 预先写定的工程门槛是「独立测试 20 回合中至少 18 个完成 20 s 行走，
并满足速度与姿态要求、无数值异常、无隐藏支撑」。实测：

| 策略 | 达到门槛的回合数 | 要求 |
|---|---|---|
| A 官方 | 0 / 20 | ≥ 18 |
| B 微调 | 0 / 20 | ≥ 18 |

（该门槛是本项目的**阶段性工程标准**，不代表临床有效性；本次小预算试训练本就不要求达到。）

因此**现阶段不应开始偏瘫损伤适应训练**：损伤带来的额外扰动会与「正常肌力下都走不到
20 s」这一混杂因素纠缠在一起，使任何侧别/辅助结论都不可解释。

### 8.2 下一次最有依据的**单项**调整

**给侧向偏移加终止条件**（`|pelvis_y − ref_y| > 0.25 m` → `terminated`）。

**依据（数字）**：

* §1.3：侧向漂移是唯一在跌倒前显著发散、且远超参考自身摆动的量（0.438 m vs 0.076 m）；
* §1.4：`drift_y` 的首次越界（3.58 s）排在跌倒（4.09 s）之前，是领先指标；
* §7.3：当前侧向类惩罚合计只有 **−15.8/步**，而存活奖励 **+100/步**，
  差 6 倍 —— 密集惩罚太弱，无法与「保高度」竞争；
  改成**终止**后，漂移的代价从「每步 −15.8」变成「损失剩余全部回报」，
  量级上与存活奖励可竞争，这正是「密集惩罚弱于存活奖励」这类问题的标准修法；
* §5.2 E1 已排除「观测裁剪」作为主因，所以不必先动观测。

**可执行命令**（只需在配置里开启该终止项，其余不动）：

```bash
# 1) 在 configs/train_healthy_v1.json 的 termination 段增加：
#    "max_lateral_dev_from_ref_m": 0.25
#    （研究环境按同一步长判定，不改变观测/动作接口）
# 2) 同时把训练预算提高到与 critic 收敛需求匹配（当前 Q 高估 36%）：
MUJOCO_GL=egl PYTHONPATH=. python scripts/train_healthy.py \
    --config configs/train_healthy_v1.json --tag healthy_v2 \
    --max-transitions 200000 --max-minutes 90
# 3) 再用固定协议评估（测试种子不变，避免据测试结果调参）：
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_healthy.py \
    --policies A=artifacts/checkpoints/LocomotionFull B=runs/train/healthy_v2 \
    --seeds 201:220 --tag healthy_vs_official_v2
```

> 注：第 2 步的 `--max-transitions`/`--max-minutes` 需要先按同样的口径扩展配置
> （`budget` 段），本次未启用超预算训练。**本次到此停止，不自动开启更大规模训练。**

**次优先项**（若侧向终止仍不足）：把 `w_survival` 从 100 降到与模仿项可比的量级
（例如 20），使「活着」不再压倒「走对」。

### 8.3 本次**没有**做的事（避免误读）

* 未加入拐杖（被动/主动均未实现）；
* 未降低任何肌力倍率（全程 1.0），未做损伤适应训练；
* 未启动临界点（辅助侧别转换点）扫描；
* 修改 `external/` 上游代码 —— **零改动**；未触碰 `env_isaaclab` 环境；
* 80/80 项断言类检查全部通过，无「验证失败」项。

---

## 9. 产物与溯源

| 产物 | 路径 |
|---|---|
| 失稳诊断 | `reports/instability_diag.json`（含 1 个种子的完整逐步序列） |
| 命名对照 | `reports/named_controls.json` |
| 资源与吞吐 | `reports/throughput.json` |
| 初始化对照 | `reports/train_check_init.json` |
| 奖励标定 | `reports/reward_calibration.json` |
| 训练前后对比 | `reports/train_vs_baseline.json` |
| 策略一致性（修正后） | `reports/verify_policy.json` |
| 试训练产物 | `runs/train/healthy_v1/`（`train_meta.json`、`train_history.json`、`run.json`；`checkpoint/` 不入库） |
| 配对评估 | `runs/healthy_vs_official/eval_report.json` + `run.json` |

每个产物的 `provenance` 都记录：项目 HEAD 与**代码补丁哈希**、上游 commit 与脏文件分类、
checkpoint 与归一化文件哈希、实际加载的模型文件哈希、依赖版本、种子、实际奖励权重与
`dynsyn_weight_amp` 固定值。
