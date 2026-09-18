# 第一阶段修复与验证 —— 验收报告

> 基线：`ab452c4`（修复前）。本报告所有数字均由**实际执行**产生，来源可查。
> 生成时间：2026-09-18。验证环境：conda `hemirl`（Python 3.12.13, mujoco 3.11.0,
> numpy 2.3.5, gymnasium 1.2.3, torch 2.14.0+cpu, SB3 2.7.1）。
>
> 一条命令可复现全部结论（含退出码传播）：
>
> ```bash
> MUJOCO_GL=egl PYTHONPATH=. python scripts/acceptance.py
> ```

---

## 0. 结论速览

| 项 | 状态 |
|---|---|
| 策略一致性验证（两条加载路径） | ✅ 已实现且验证通过（`max|Δa| = 5.96e-08` ≤ `1e-5`，126 个样本） |
| 统一的研究环境（训练/评估共用） | ✅ 已实现且验证通过（11 项回归 + 环境接口检查） |
| 终止/计时/统计口径修复 | ✅ 已实现且验证通过（含终止步记录、实际仿真时间、三分语义） |
| 物理量修正（根节点世界角速度） | ✅ 已实现且验证通过（有限差分 + Jacobian 双重校验） |
| 肌力验证修正（V6 重写） | ✅ 已实现且验证通过（22/22） |
| 约束力分解（接触/限位/equality） | ✅ 已实现且验证通过（分解残差 2.3e-13） |
| 可复现性 | ✅ 全部实验在干净 commit `9326e19` 下重跑，指标**逐位相同** |
| 短时基线复验（5 seed） | ✅ 完成，与旧记录一致（差异已解释） |
| 长时评估（20 s，5 seed） | ⚠️ **全部跌倒**（4.46–4.90 s），如实报告，未做任何隐藏支撑 |
| 肌力扫描（左右各 5 seed） | ✅ 完成，走完/跌倒分开统计 |
| 实验溯源补齐 | ✅ 已实现（HEAD / dirty 补丁哈希 / 上游脏文件分类 / 模型文件哈希） |
| 损伤适应训练 | ❌ **未实现**（本次范围外） |

---

## 1. 优先修复项：策略一致性验证

### 1.1 已提交的 `reports/verify_policy.json` 是修复前的旧结果

审查意见指出 `max_abs_action_diff ≈ 1.97314` 与脚本标准 `1e-5` 冲突。核查结论：**该 JSON 确为修复前产物**。

| | 值 |
|---|---|
| 提交版本（`4df6953`）中的 `max_abs_action_diff` | `1.9731438159942627` |
| 当前代码实测 | `5.960464477539063e-08` |

两条证据：

1. **重跑**：用**未修改**的旧脚本复跑，得到 `5.96e-08`（通过），与提交值不符 → 提交的是旧结果。
2. **缺陷复现**：新验证模块显式关掉「`latent_pi` 尾部 ReLU」后可复现 `max|Δa| = 1.9763`，
   与旧值 1.9731 吻合；且多层探针把最早差异定位在 **`latent_pi`** 这一层：

```
[layer] stage_diffs={'latent_pi': 0.0, 'mu': 0.0, 'tanh_mu': 0.0, 'final_action': 0.0}
[defect] 缺尾部 ReLU 时 first_divergent_stage = 'latent_pi', max|Δa| = 1.9763
```

根因：SB3 的 `create_mlp(features_dim, -1, net_arch, activation_fn)` 在 `output_dim <= 0` 时
**最后一层之后也会加激活**，所以 `latent_pi` 以 ReLU 结尾。早期实现漏掉它。

> 旧 JSON 已原样保存在 `reports/verify_policy_pre_fix.json`，作为历史证据，未被覆盖删除。

### 1.2 重写后的验证（`hemirl/policy_verify.py` + 薄封装 `scripts/verify_policy.py`）

要求逐条落实：

| 要求 | 实现 |
|---|---|
| 同一 checkpoint、同一归一化观测、确定性推理 | 两条路径吃**同一个** `VecNormalize.normalize_obs` 输出数组 |
| 核查网络层/激活 | `latent_pi_layers` 落盘为 `[Linear(1024,3601), ReLU, Linear(1024,1024), ReLU, Linear(1024,1024), ReLU]` |
| 核查组展开索引 | `build_group_expansion` 与上游 `DynSynLayer.repeat_replace_x` 语义对照，138 组覆盖 700 块肌肉、无重叠 |
| 核查动作缩放 | `tanh(mean) → 按组索引展开 → clip[-1,1]`；断言 `group_value_repeat_ok`、`in_range` |
| 核查 `dynsyn_weight_amp` 语义 | 用两条**可执行事实**固定：`amp=None` 与 `amp=0` 输出**完全相同**（后者按公式恒为 weight=1），`amp=0.05` **必然不同**（证明重标定分支是活的） |
| 多 reset 观测 + 行走轨迹观测 | 6 个 reset（seed 1000+）+ 120 个真实行走轨迹观测（seed 0/1/2），共 **126** 个样本 |
| 官方基线语义不变 | 官方基线只用 `amp=None`；`amp=0.05` 只写入 `named_experiments`，不参与判定 |
| 失败返回非零退出码 | `passed=False` → `sys.exit(1)`；已实测 |
| 记录 passed/阈值/最大误差/样本数/代码版本/checkpoint 哈希 | 全部写入 `reports/verify_policy.json` |
| 加入验收入口 | `scripts/acceptance.py` 第 4 步（检测到 checkpoint 才运行） |

结果：**7/7 检查通过**，`max|Δa| = 5.960e-08`，`code_version = 9326e19`，
`best_model.zip sha256 = 0a3607c0a9d745a6…`。

---

## 2. 统一回合终止、计时与研究环境

新增 `hemirl/research_env.py`：`ResearchEnvConfig` / `RewardSplitConfig` /
`ResearchLocomotionEnv` / `build_research_env()` / `reference_continuity_report()`。
**训练与评估共用同一个工厂**（`build_research_env`），语义不再只存在于评估循环里。

### 2.1 语义三分（已实测）

| 类别 | 判据 | `termination_source` | 返回 |
|---|---|---|---|
| 物理跌倒 / 明确任务失败 | 骨盆高度 < 0.55 m；直立偏差 > 60°；根速度/关节速度范数超限 | `physical_fall` | `terminated=True` |
| 达到规定评估时长 / 本地上限 | `sim_time >= time_limit_s`；或达到 `max_control_steps` | `time_limit` / `step_cap` | `truncated=True` |
| 数值异常 | qpos/qvel/qacc/act 出现 NaN/Inf | `numeric_anomaly` | `terminated=True`，**且 `physical_fall=False`** |

数值异常与生理性跌倒分开统计：`termination_metadata()` 同时给出
`numeric_anomaly` 与 `physical_fall` 两个互斥布尔位，汇总表的 `outcome_counts`
也把两者分开计数。

### 2.2 逐条要求对照

| 要求 | 实现与证据 |
|---|---|
| 1. 记录每一个已执行的物理步，含终止步 | `StepRecord` ledger 覆盖全部控制步；`ledger_covers_terminating_step=True`，`all_steps_recorded=True`；终止步的 `pelvis_z/up_tilt_deg/qpos_track_err` 都是实测值 |
| 2. 用实际仿真时间差算运行时间 | `elapsed_time_s = data.time - reset_time`；`time_source` 字段显式标注；`per_step_gap_max_dev ≈ 1e-16` |
| 3. 显式记录 reason/source/elapsed/实际控制步数 | `termination_metadata()` 一次给全，并写入每条 episode 结果与 ledger |
| 4. 研究模式覆盖官方参考误差终止与官方时间截断 | `suppresses_official_reference_termination/time_truncation = True`；官方标志仍被计数到 `n_official_terminated_flags`（覆盖但留证据） |
| 5. 本地步数上限必须有明确截断标志 | `reached_step_cap` / `reached_time_limit` 布尔位 |
| 6. 终止后不得继续推进或自动 reset | `step()` 在 done 后抛 `RuntimeError`；实测计数不变 |
| 7. 参考跟踪误差仅作描述指标 | 字段名 `mean/max/final_qpos_track_err`，文档与报告均标注「不作为终止依据、不代表平衡能力」 |
| 8. 奖励拆分 | `RewardSplitConfig`：`imitation` / `energy` / `survival_physical`（**不依赖参考**）/ `official_healthy`（依赖参考，单独保留）；`reward_mode='official'` 时返回值与上游逐位相同，`'split'` 时才改变返回值 |
| 保持 obs/action 接口 | 未新增任何观测；动作仍经官方 `MuscleNormWrapper`；`action`/`observation_space` 维度与 checkpoint 一致 |

### 2.3 修复的口径缺陷（附实测）

| 缺陷 | 修复前 | 修复后 |
|---|---|---|
| 官方终止在记录前 `break` | 只记录 **175** 步 | 记录 **176** 步（含终止步） |
| 运行时间用「步数 × 标称 dt」 | 3.50 s | **3.52 s**（`data.time` 差） |
| 官方 `terminated` 被统一解释为参考偏差 | 无法区分时间上限 | `official_time_limit` / `official_reference_deviation` 分开 |
| 研究模式仍受官方 3.51 s 截断 | 是 | 由 `max_episode_seconds` 取代 |
| 研究终止逻辑只存在于评估循环 | 是 | 提升为环境对外行为，训练可直接复用 |
| 步数上限用 `round` 早于时间上限触发 | 175 步 / `step_cap` | 改为**向上取整** → 176 步 / `time_limit`（与官方口径一致） |

### 2.4 回归测试（`tests/test_research_env.py`，11/11 通过）

每条测试对应一个真实出错的点，而非复述实现：T1 终止步被记录、T2 时间截断与步数上限标志、
T3 计时来源、T4 物理跌倒、T5 官方参考终止在研究模式被覆盖但仍留证据、
T6 数值异常单独归类、T7 终止后不得推进、T8 奖励拆分、T9 根速度 Jacobian、
T10 约束力分解、T11 参考连续性。

---

## 3. 物理量、肌力验证与文档修正

### 3.1 根节点速度改用旋转 Jacobian

`root_state()` 现在用 `mujoco.mj_jacBody(qpos) @ qvel` 给出**世界系**线速度与角速度，
旧式「槽位重排」的结果同时被计算并记录差值。校验方式：

* 线速度：与 `data.xpos[pelvis]` 的中心差分一致（`atol=1e-5`）；
* 角速度：与 `Ṙ Rᵀ`（反对称阵）分量一致（`atol=1e-5`）。

| 方法 | 线速度最大偏差 | 角速度最大偏差 |
|---|---|---|
| 旧索引写法（策略驱动，80 步） | `1.11e-16` | **`0.154` rad/s** |
| 旧索引写法（跌倒段） | `4.44e-16` | **`1.224` rad/s`** |
| 整段 episode（5 seed 短时基线） | — | **`0.2005` rad/s** |

结论：旧写法的**线速度**在该模型上恰好等价，但**角速度不等价**。
研究终止判据中的 `max_root_ang_speed` 与该速度的**报告值**都已改用 Jacobian。

### 3.2 肌力验证 V6 重写（`reports/verify_strength.json`，22/22 通过）

原 V6 通过「放大初始关节角」改姿态，四组基准力完全一致 —— 说明那些肌肉**长度根本没变**，
结论是空的。现在：

1. 按**关节名**选肩（`elv_angle_r` / `shoulder_elv_r` / `shoulder_rot_r`）与肘（`elbow_flexion_r`），
   偏移量都落在 `jnt_range` 内；
2. **先证明长度确实改变**：肩姿态下目标肌肉最大长度变化 0.0234 m，肘姿态下 0.0313 m；
3. **再用 $F_{active}(a) = F(a) - F(0)$ 分离主动分量**，验证缩放比恒等于倍率：

| 姿态 | 测到的肌肉数 | $F_{active}$ 缩放比 mean / min / max |
|---|---|---|
| `shoulder_elv` (+0.60 rad) | 61 | 0.5 / 0.5 / 0.5 |
| `elbow_flex` (+1.00 rad) | 61 | 0.5 / 0.5 / 0.5 |
| `combined` | 61 | 0.5 / 0.5 / 0.5 |

4. 被动力对照保留：`active_only` 下被动力**逐元素不变**；`active_and_passive` 下按倍率缩放
   （ratio mean = 0.5），两种模式可区分（最大差 0.82 N）。

5. **删除了 `q[3:7] = key[3:7]` 这类索引假设**：本模型没有 freejoint
   （`nq == nv == 85`，骨盆是 3 slide + 3 hinge），不存在「根节点四元数」。

### 3.3 `AGENTS.md` 的 F0 槽位说明已修正

原文写作「F0 … 在 `gainprm[0]`」，与实现和实测冲突。已改为：

> **F0**：在 `gainprm[2]`（主动通道）与 `biasprm[2]`（被动通道），不是 `gainprm[0]`。
> 该结论由扰动–响应实证得出，实现见 `hemirl/muscle_actuator.py:identify_f0_slots`。

### 3.4 力学解释不再混淆四类力

新增 `hemirl/forces.py`：

* `generalized_force_terms()`：给出 `qfrc_actuator / qfrc_passive / qfrc_bias / qfrc_constraint /
  qfrc_applied / xfrc_applied / qfrc_inverse / qacc` 的范数，并**显式附带单位警告**：
  广义力混合 N 与 N·m，范数只用于量级/突变检测，**不能**解释为承重或贡献占比。
* `decompose_constraint_forces()`：按 `mjCNSTR_*` 把 `qfrc_constraint` 拆成
  接触 / 限位 / equality / 摩擦，用「求和后与 `data.qfrc_constraint` 比对」的数值方法确定符号约定。

实测（策略驱动 80 步，范数均值）：

| 类别 | 均值范数 |
|---|---|
| equality | 1879.2 |
| contact | 837.5 |
| limit（关节/腱限位） | 621.1 |
| friction | 0.0 |
| **分解残差** | **2.27e-13** |

因此不再出现「把 `qfrc_constraint` 整体归因于 equality」的表述：
接触与限位都在实质受力，报告里给的是**合计量 + 分解量**。

* `muscle_active_passive_split()`：$F_{passive}=F(act{=}0)$，$F_{active}=F(act)-F(act{=}0)$。
  策略驱动下 `mean|F_active| = 28.12 N`、`mean|F_passive| = 1.22 N`（`abs_max` 分别为 771 N / 221 N）。

---

## 4. 基线与长时评估

### 4.1 A. 短时基线复验（官方配置，seed 0–4）

命令：
```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
  --episodes 5 --seed 0 --termination official --tag eval_short_baseline_official
```

| 指标 | 旧记录（修复前） | 本次 | 差异解释 |
|---|---|---|---|
| 步数 | 175 | **176** | 终止步现在也被记录 |
| 运行时间 | 3.50 s | **3.52 s** | 来自 `data.time` 差，且多一步 |
| 平均速度 | 0.960 ± 0.013 m/s | **0.9582 ± 0.0131 m/s** | 分母口径变化 |
| 位移 | 3.358 m | **3.372 m（前进 x）** | 同上；并新增侧向分量 0.0835 m |
| 最低骨盆高度 | 0.875 m | **0.8748 m** | 一致 |
| 官方终止触发 | 0/5（均为时间上限） | **0/5 参考偏差，5/5 时间上限** | 语义现在被显式区分 |

> **口径差异的定量证明**：修复后的研究环境在**修正 `ceil` 之前**曾在 175 步被 `step_cap` 截断，
> 那次运行的数值为 `0.9595 / 0.9729 / 0.9627 / 0.9248`（上肢扫描），
> 与旧记录**逐位相同**。这直接说明：本次修复**没有改变动力学**，只改变了记录与终止口径。

### 4.2 B. 研究环境长时评估（20 s，seed 0–4）

前置检查（`scripts/probe_reference_continuity.py`，`reports/reference_continuity.json`）：

| 项 | 值 | 含义 |
|---|---|---|
| `terminate_time_s` / `num_frames` / `framerate_hz` | 1.17 / 117 / 100 | 周期定义 |
| `frame_span_s` | 1.16 | 帧序列实际覆盖时长 |
| `wrap_frame_gap_s` | **0.01** | 因 `clip(…, 0, num_frames-1)`，每周少一帧 |
| `max_step_change_within_cycle` | 0.0848 | 周期内单控制步参考变化上限 |
| `max_step_change_at_cycle_boundary` | 0.0677 | 边界处的单步变化 |
| `boundary_over_within_ratio` | **0.80** | < 1 → 边界跳变比周期内正常步**更小**，无异常跳变 |
| `pose_wrap_discontinuity_max_rad` | 0.0329 | 轨迹**非严格周期**，姿态在边界不闭合 |
| `per_cycle_forward_from_qpos_m` | 1.1946 | 与 `xpos` 推出的每周期前进量**完全一致** |
| `forward_accumulation_rel_err` | 4.0e-4 | 前进位移累积正确 |
| `needs_loop_guard` | **False** | 不需要额外护栏 |

结论：参考循环**连续、无异常跳变、前进累积正确**；但**非严格周期**（每周 0.0329 rad 姿态缺口、
0.01 s 帧缺口），20 s 内累计约 0.66 rad 姿态缺口与 0.20 m 位移缺口，属于需要如实说明的既有特性。
上游另有一处既有行为：`ref_time = data.time + init_time + dt`，即**参考相对状态超前 1 个控制步**
（0.02 s）；为保持 checkpoint 接口一致未做修改。

评估结果（**如实报告，未做任何中途 reset、根节点覆盖或稳定力补偿**）：

| seed | 步数 | 存活 | 终止来源 | 终止原因 |
|---|---|---|---|---|
| 0 | 223 | 4.46 s | `physical_fall` | 骨盆直立偏差 62.0° > 60.0° |
| 1 | 245 | 4.90 s | `physical_fall` | 骨盆直立偏差 60.7° > 60.0° |
| 2 | 226 | 4.52 s | `physical_fall` | 骨盆直立偏差 61.6° > 60.0° |
| 3 | 224 | 4.48 s | `physical_fall` | 骨盆直立偏差 62.1° > 60.0° |
| 4 | 236 | 4.72 s | `physical_fall` | 骨盆直立偏差 60.6° > 60.0° |

汇总：存活 **4.62 s（4.46–4.90）**，计划时长 20 s；`outcome_counts = {physical_fall: 5}`；
侧向漂移 0.401–0.521 m（短时基线仅 0.084 m）；最低骨盆 0.686 m；参考跟踪误差 mean 0.0594 / max 0.2987。

**失败归因（基于证据，不是猜测）**：

1. **不是参考循环造成的**。轨迹分析显示周期边界处 `max|Δqpos_ref|` 为 0.037–0.060，
   而周期内 95 分位为 0.080 → **边界跳变小于正常单步变化**；
   参考跟踪误差在 3 s 前平稳（0.028–0.047），到跌倒瞬间才升到 ~0.30。
2. **主因是策略自身的长时稳定性**。所有种子的侧向漂移**同向（+y）**且是**晚期快速发散**：
   `|y| > 0.2 m` 首次出现在 4.18–4.68 s，即跌倒前不到 0.3 s。骨盆高度在跌倒前仍 > 0.66 m。
3. 该 checkpoint 的官方配置**只在 3.51 s（3 个步态周期）内做过终止**，超出该窗口的行为
   从未被检验；4.6 s 已跨过 3 个周期边界，说明这不是单次循环边界事件引发的偶发失败。

**未采取的「作弊」手段**：没有中途 reset、没有覆盖根节点、没有加稳定力、没有重训。
视频证据：`runs/eval_long_research_video/rollout.mp4`（224 帧 / 50 fps / egl），
关键帧在 `reports/video_frames_long/`（`frame_223.png` 可直接看到倒地姿态）。

### 4.3 C. 最小肌力复验（原短时协议，配对照种子，左右各 5 seed）

命令：
```bash
for SIDE in R L; do
  MUJOCO_GL=egl PYTHONPATH=. python scripts/strength_sweep.py \
    --paretic-side $SIDE --seeds 0 1 2 3 4 --termination research \
    --tag sweep_${SIDE}_short_research
done
```

**患侧 R**（`runs/sweep_R_short_research/`）

| 扫描 | 倍率 | 走完 / 跌倒 | 存活 | 前进 x | 侧漂 y | v(走完) | v(跌倒) |
|---|---|---|---|---|---|---|---|
| A 上肢 | 1.0 | 5 / 0 | 3.52 s | 3.372 m | +0.084 | 0.9582 | — |
| A 上肢 | 0.75 | 5 / 0 | 3.52 s | 3.419 m | +0.087 | 0.9717 | — |
| A 上肢 | 0.5 | 5 / 0 | 3.52 s | 3.381 m | +0.111 | 0.9611 | — |
| A 上肢 | 0.25 | 5 / 0 | 3.52 s | 3.248 m | +0.122 | 0.9234 | — |
| B 下肢 | 1.0 | 5 / 0 | 3.52 s | 3.372 m | +0.084 | 0.9582 | — |
| B 下肢 | 0.75 | 5 / 0 | 3.52 s | 3.250 m | −0.009 | 0.9232 | — |
| B 下肢 | 0.5 | **0 / 5** | 1.80 s | 1.938 m | −0.061 | — | **1.0924** |
| B 下肢 | 0.25 | **0 / 5** | 0.74 s | 0.713 m | −0.053 | — | **0.9735** |

**患侧 L**（`runs/sweep_L_short_research/`）

| 扫描 | 倍率 | 走完 / 跌倒 | 存活 | 前进 x | 侧漂 y | v(走完) | v(跌倒) |
|---|---|---|---|---|---|---|---|
| A 上肢 | 1.0 | 5 / 0 | 3.52 s | 3.372 m | +0.084 | 0.9582 | — |
| A 上肢 | 0.75 | 5 / 0 | 3.52 s | 3.284 m | +0.109 | 0.9335 | — |
| A 上肢 | 0.5 | 5 / 0 | 3.52 s | 3.211 m | +0.137 | 0.9133 | — |
| A 上肢 | 0.25 | 5 / 0 | 3.52 s | 3.069 m | +0.216 | 0.8744 | — |
| B 下肢 | 1.0 | 5 / 0 | 3.52 s | 3.372 m | +0.084 | 0.9582 | — |
| B 下肢 | 0.75 | 5 / 0 | 3.52 s | 3.456 m | +0.125 | 0.9824 | — |
| B 下肢 | 0.5 | **1 / 4** | 2.76 s | 2.289 m | +0.112 | 0.8924 | 0.8560 |
| B 下肢 | 0.25 | **0 / 5** | 1.14 s | 0.852 m | +0.254 | — | **0.7838** |

要点：

* **下肢损伤是主因**：下肢 0.5 时 5 个种子基本全倒，0.25 时平均存活不到 1 s；
  上肢降到 0.25 仍能走完全部 3.52 s（速度仅下降 3.6%，R 侧 0.9582 → 0.9234）。
* **跌倒前的速度不是效果改善**：R 下肢 0.5 时跌倒样本 `v = 1.0924 m/s`，**高于**基线 0.9582 m/s，
  但这只说明它「跌之前冲得快」（存活仅 1.80 s）。汇总里 `speed_mean_completed_only_mps`
  与 `speed_mean_fallen_only_mps` 分开给出，并在 `summary.json` 中附 `interpretation` 说明。
* **左右不完全对称**：下肢 0.5 时 R 为 0/5 走完、L 为 1/5 走完（L 平均存活 2.76 s）。
  可能来自参考动作本身的左右不对称与单条参考轨迹；样本仅 5 个种子，
  **不应过度解读**，记录在案供下一阶段用更多种子确认。
* 与修复前旧记录的一致性：R 下肢 0.5 → 1.80 s、0.25 → 0.74 s，旧记录为 1.80 s / 0.74 s，
  **完全一致**；上肢三档速度与旧记录逐位相同（见 4.1 的 `ceil` 证据）。

---

## 5. 实验溯源（第七节）

`provenance.run_provenance()` 现在每次实验都会落盘：

| 要求 | 字段 |
|---|---|
| 本项目 HEAD | `code_version.commit` / `branch` / `remote` |
| 工作区 dirty 状态 | `code_version.dirty`（任意改动）与 `code_version.code_dirty`（仅代码） |
| 未提交修改的补丁/快照哈希 | `code_version.patch_sha256`（`git diff HEAD -- hemirl scripts tests configs` 的 SHA-256）、`patch_sha256_all`（全量）、`describe`（如 `9326e19+code-dirty:1f3c9a7e`）、`untracked_files` |
| 上游 commit | `upstream.{MS-Human-700,msgym}.commit` |
| 上游脏文件列表，区分符号链接与源码 | `upstream.*.dirty_raw`、`symlink_changes`、`source_changes`、`n_source_changes` |
| checkpoint 与归一化文件哈希 | `checkpoint.model_zip_sha256`（`0a3607c0…`）、`env_zip_sha256`（`e359b32f…`） |
| 实际加载的模型路径与关键文件哈希 | `loaded_model_file`（路径、是否符号链接、解析后路径、sha256、大小） |
| 配置 / 依赖版本 / 种子 / 推理设置 / 实际肌力倍率 | `extra.research_env_config`、`dependencies`、`seeds`、`policy_inference`、`strength` |

**一处设计取舍（值得记录）**：`describe` 只根据 `hemirl/`、`scripts/`、`tests/`、`configs/`
这四个路径判断「代码是否变脏」。原因是运行本身会改写 `reports/` 与 `runs/`，
若把它们计入，**每个结果都会带上 `+dirty` 而失去标识意义**。
输出文件的变动仍完整记录在 `other_dirty_files` 中。
因此本报告下的全部结果都带 `code_version = 9326e19`（干净代码快照）。

全部实验在干净提交 `9326e19` 下重跑过一次，与先前的运行结果**逐位相同**
（A 短时基线的 `steps/sim_time/speed/前向位移/最低骨盆`，B 长时的 `steps/sim_time/侧漂/最大倾角`，
C 两侧扫描的 16 个 `(扫描, 倍率)` 组合的走完数与速度，全部 `identical=True`），
即该评估链路在本环境下是**确定性可复现**的。

旧结果**未被覆盖**：`eval_min_official`、`eval_repeats_research`、`eval_video`、
`eval_R_upper0.5_bothmodes`、`sweep_L_research`、`sweep_R_research` 全部保留；
新实验写入
`eval_short_baseline_official`、`eval_long_research`、`eval_long_research_video`、
`sweep_R_short_research`、`sweep_L_short_research`。

---

## 6. 交付清单

| # | 交付物 | 位置 |
|---|---|---|
| 1 | 代码修复 + 回归测试 | `hemirl/policy_verify.py`、`hemirl/research_env.py`、`hemirl/forces.py`、`hemirl/rollout.py`、`hemirl/termination.py`、`hemirl/provenance.py`、`hemirl/render.py`、`tests/test_research_env.py`、`tests/test_core.py` |
| 2 | 官方复现环境与研究环境构建入口 | `hemirl/envs.py:build_env()`（官方）、`hemirl/research_env.py:build_research_env()`（研究，训练/评估共用） |
| 3 | 一条可执行验收命令 | `MUJOCO_GL=egl PYTHONPATH=. python scripts/acceptance.py`（退出码已实测传播） |
| 4 | 新生成的验证/基线/长时/扫描结果 | `reports/acceptance.json`、`reports/verify_policy.json`、`reports/verify_strength.json`、`reports/dynamics_audit.json`、`reports/reference_continuity.json`、`reports/unit_tests.json`、`runs/eval_short_baseline_official`、`runs/eval_long_research`、`runs/eval_long_research_video`、`runs/sweep_{R,L}_short_research` |
| 5 | 更新后的 README 与验收报告 | `README.md`、本文件 |

---

## 7. 状态逐项分类（任务书要求）

### ✅ 已实现且实际验证通过

1. 策略一致性验证（7/7，`max|Δa| = 5.96e-08`，126 样本，含行走轨迹观测，失败返回非零退出码）。
2. 统一研究环境（`build_research_env`），终止语义三分、终止步记录、实际仿真时间、
   终止后拒绝推进、本地截断标志、奖励拆分（11/11 回归测试）。
3. 根节点世界角速度改用旋转 Jacobian（有限差分 + Jacobian 双重校验；
   旧写法角速度偏差实测 0.154–1.224 rad/s，已量化）。
4. 肌力验证 V6 重写（长度变化先证后测；$F_{active}$ 缩放比 0.5/0.5/0.5；被动通道对照；22/22 通过）。
5. 约束力按类型分解（残差 2.27e-13），报告不再把 `qfrc_constraint` 归因于 equality。
6. 肌肉力主动/被动分离（`F_active`/`F_passive` 定义自洽，误差 0）。
7. 短时基线复验（5 seed）、右侧与左侧肌力扫描（各 40 episode）、长时 20 s 评估（5 seed）+ 视频。
8. 实验溯源补齐（含 dirty 补丁哈希与上游符号链接/源码变化分类）。
9. `AGENTS.md` F0 槽位说明修正；参考连续性前置检查脚本。

### ⚠️ 已实现但受限制未能完全验证（如实标注）

1. **长时评估未通过**：官方策略在 4.46–4.90 s 全部跌倒。这是**结论**而不是未完成项，
   但它意味着「正常肌力 20 s 稳定行走」这一前提**不成立**，下一阶段的损伤适应训练
   必须先解决或绕过它（见第 8 节）。
2. **左右对称性只有 5 个种子**：下肢 0.5 的 R/L 差异在统计上不足以定论。
3. **数值异常**：端到端测试通过在 `step` 后注入 NaN 的方式触发（仿真器本身未自发产生 NaN）；
   真实 NaN 的产生路径尚未观察到。
4. **`eval_long_research` 只保存了 1 段视频**（seed 0）；其余 4 个种子只有轨迹 npz（不入库）。

### ❌ 验证失败

无。所有断言类检查（18 核心 + 11 环境回归 + 22 肌力 + 7 策略 + 3 验收项）全部通过。

### ⛔ 尚未实现（本次范围外，明确声明）

1. **损伤适应训练**（本报告全部内容均为**固定策略**评估，**不是**偏瘫适应训练结果）。
2. 拐杖机器人（被动/主动均未实现）。
3. 辅助侧别转换点的检验（既未证真也未证伪 —— 因为还没有可训练的损伤适应策略）。
4. 新增观测（如辅助工具状态、患侧代偿指标）——按约定只能放在后续独立配置里，本次未做。
5. 真实卒中患者模拟（本项目始终是**模型化肌力下降**，不是患者数据）。

---

## 8. 下一阶段：损伤适应训练需要什么

### 8.1 用哪个环境

**必须**通过研究环境工厂构建，才能保证训练与评估语义一致：

```python
from hemirl.research_env import ResearchEnvConfig, RewardSplitConfig, build_research_env
from hemirl.termination import TerminationConfig

env, raw = build_research_env(
    env_cfg=ResearchEnvConfig(
        termination=TerminationConfig(kind="research"),   # 覆盖官方参考误差终止
        max_episode_seconds=8.0,                          # 训练时长（见 8.3 的阻塞）
        reward_mode="split",                              # 物理存活与模仿分开返回
        reward_split=RewardSplitConfig(w_survival=100.0, include_official_healthy=False),
    )
)
```

**不要**直接用 `gym.make("msgym/LocomotionFullEnv-v1")` 训练：那会把「偏离参考即终止」
和「3.51 s 硬上限」带进训练，训练出的策略无法与本阶段的评估口径比较。

### 8.2 用哪些初始化权重

* **起点**：官方 DynSyn-SAC checkpoint
  （`artifacts/checkpoints/LocomotionFull/checkpoint/best_model.zip`，
  `sha256 = 0a3607c0a9d745a6bb5a5ffceb7539a7405d455943abf283d4f8107f22a47829`）
  **`actor` 全部参数**（`latent_pi` / `mu` / `log_std` / `dynsyn_layer`，138 肌群输出）+ **`VecNormalize` 统计**
  （`best_env.zip`，`sha256 = e359b32fa026a02746cffcba6ee3ef9cab4389d37c74b19bd21caa0f3abc0086`）。
* **critic 可以重置**：SAC 的 critic 与 `log_alpha` 需要重建（肌力条件会改变价值函数）；
  官方 `alpha` 需要按新的奖励量级重新标定（`reward_mode='split'` 的默认量级与官方不同）。
* **观测/动作接口必须保持不变**（3601 维 obs / 700 维肌肉动作），
  否则上面的权重无法直接载入。新增观测只能放进**独立配置**并从头训练。

### 8.3 尚存的具体阻塞

1. **长时基线不成立（最关键）**。官方策略 4.6 s 就侧向失稳跌倒，比计划的训练回合（8–20 s）短得多。
   在解决之前，任何「损伤适应训练」都会与「长时稳定性」这一混杂因素纠缠。可选路径：
   (a) 以更短的回合（≤ 2 个周期，约 2.3 s）先做损伤适应，
   (b) 先用域随机化/课程学习把正常肌力的长时稳定性练上来再做损伤，
   (c) 显式把侧向漂移作为额外的终止/惩罚项。
   三种路径都会改变与官方基线的可比性，需要明确记录为**命名实验**。
2. **参考轨迹非严格周期**：每周 0.0329 rad 姿态缺口 + 0.01 s 帧缺口（`wrap_frame_gap_s`），
   20 s 累计约 0.66 rad / 0.20 m。回合越长影响越大。上游只读，因此只能在训练配置里
   选择更短回合，或后续用**自己生成**的严格周期轨迹（需要另做并单独标注）。
3. **参考超前 1 个控制步**（`ref_time = time + init_time + dt`）：这是上游既有行为，
   为兼容 checkpoint 未修改；若将来重新训练，应把「是否保留该超前」作为一个显式配置项。
4. **奖励量级**：`reward_mode='split'` 的总量与官方不同（官方 `w_healthy=100` 依赖参考姿态，
   拆分后 `survival_physical=100` 不依赖参考），需要重新标定 `alpha` 与各权重；
   配置必须随运行落盘（已由 `research_env_config` 覆盖）。
5. **左右对称性样本不足**：只有 5 个种子，R/L 下肢 0.5 的差异不能定论。
6. **计算资源**：CPU 版 torch，单步约 20 ms；尚未评估 GPU 训练可行性。
7. **上游依赖复原**：`external/` 与 `artifacts/` 不入 git，
   需要 `scripts/setup_external.sh` + Release 下载 checkpoint 才能在新机器复现。

### 8.4 本阶段**没有**做的事（避免误读）

> 本报告的全部结果都是**固定官方策略权重**在不同肌力设定下的短时/长时表现。
> **没有**进行任何损伤适应训练，因此**不能**推断「患侧拄拐更好」或「存在辅助侧别转换点」。
> 临界点仍然是**待检验假设**；本阶段只是把检验它所需的环境、口径与证据链准备好。
