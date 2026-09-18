# Phase 1 阶段报告 — MS-Human-700 + msgym 正常行走复现与患侧肌力参数化
> **历史记录提示（2026-09-18 追加）**：本报告的运行时间 / 步数使用**修复前**的记录口径
> （终止步未被记录：175 步 / 3.50 s），且 `reports/verify_policy.json` 当时保存的是修复前的
> 中间结果（`max_abs_action_diff = 1.9731`）。修复后的口径与验收结果见
> [`reports/phase2_acceptance_report.md`](phase2_acceptance_report.md)。
> 本文内容**保持原样**作为历史记录，未做改写。
> 生成日期：2026-09-16
> 工作区：`/home/zrq/Documents/muscle-rl`（`muscle-pt` 项目的独立新增后端，未覆盖原实现）
> 环境：conda `hemirl`（Python 3.12.13，mujoco 3.11.0，gymnasium 1.2.3，stable_baselines3 2.7.1，torch 2.14.0+cpu，numpy 2.3.5）

---

## 0. 溯源（provenance）

| 项 | 值 |
|---|---|
| 官方模型仓库 | `https://github.com/LNSGroup/MS-Human-700` @ `2d686957aefd5739cf4d2859a6acfd94c8f84400`（`external/MS-Human-700`） |
| 官方 RL 仓库 | `https://github.com/LNSGroup/msgym` @ `ad3aac166dab3577d7d16b4a03faccef94c7aac7`（`external/msgym`） |
| checkpoint 来源 | GitHub Release `Checkpoints` → `LocomotionFull.zip` |
| `best_model.zip` SHA-256 | `0a3607c0a9d745a6bb5a5ffceb7539a7405d455943abf283d4f8107f22a47829` |
| `best_env.zip` SHA-256 | `e359b32fa026a02746cffcba6ee3ef9cab4389d37c74b19bd21caa0f3abc0086` |
| `MS-Human-700.xml` SHA-256 | `755124766e0e4ed9ddac60c8c4f505f94a6adaa2d51665108eee0cb1f677681d` |
| 原 `muscle-pt` 仓库 | `/home/zrq/Documents/muscle` @ `c89c7976ada19b5130ee3e7ae208347fe9a48be4`（**未改动**） |

每次运行都会落盘同结构 provenance：`runs/<tag>/run.json`。

---

## 1. 代码核查结论（以当前代码为准）

### 1.1 模型结构（`reports/model_inspection_full.json`）

| 项 | 实测值 |
|---|---|
| body / joint / **qpos=nv** / actuator | 81 / 85 / 85 / **700** |
| 肌肉激活状态数 `na` | 700（与 `nu` 一一对应） |
| geom / site / tendon | 349 / 2856 / 700 |
| equality 约束 | 42（**全部** `mjEQ_JOINT`） |
| 传感器 | 0 |
| timestep / frame_skip | 0.002 s（500 Hz）/ 10 → **控制周期 0.02 s（50 Hz）** |
| 积分器 / 求解器 / 接触锥 | Euler / Newton(100 iter) / pyramidal |
| `option.disableflags` | 0（**没有任何物理项被禁用**） |

**根节点自由度**：`pelvis` 由 **6 个独立关节**（3 slide + 3 hinge）构成，**不是 freejoint**。

> ⚠️ **重要实测纠正**：`pelvis` body 的坐标系相对世界系绕 x 轴旋转了 −90°，因此**关节名与实际世界方向不一致**（`scripts/probe_root_layout.py`，`reports/root_layout.json`）：
>
> | qpos 槽位 | 关节名 | **实际世界效果** |
> |---|---|---|
> | `[0]` | `pelvis_tz` | 世界 **−y**（侧向） |
> | `[1]` | `pelvis_ty` | 世界 **+z（竖直）** |
> | `[2]` | `pelvis_tx` | 世界 **+x（前进）** |
> | `[3]` | `pelvis_tilt` | 绕世界 y（俯仰） |
> | `[4]` | `pelvis_list` | 绕世界 x（侧倾） |
> | `[5]` | `pelvis_rotation` | 绕世界 z（偏航） |
>
> 竖直方向是 `qpos[1]`。同理 pelvis 的局部 z 轴指向世界 −y（水平），所以在全零姿态下它与世界 z 轴夹角是 90°，**不能**直接当作"直立轴"——终止判据里的直立轴因此按参考姿态标定（见 §5.2）。

**肌肉执行器**（全部 700 个）：`<general class="muscle" tendon=... lengthrange="lo hi" gainprm="0.75 1.05 F0" biasprm=...>`；
编译后 `dyntype/gaintype/biastype` 全为 MUSCLE（4/2/2），`ctrlrange=[0,1]`，`gear=1`，**没有额外的 motor 或 position actuator**。

**被动参数**（影响运动，必须在审计中列出）：

| 项 | 非零自由度 | 取值 |
|---|---|---|
| `dof_damping` | 79 / 85 | 0.01, 0.05, 0.1, 0.3, 0.5, 0.6, 1.0, 5.0（髋 1.0 / 膝 0.5 / 踝 0.1 / 脊柱与肩胛带 5.0） |
| `dof_armature` | 79 / 85 | 1e-4, 0.01, 0.1 |
| `dof_frictionloss` | **0 / 85** | 无关节摩擦 |
| `jnt_stiffness` | **35 / 85** | 全部 = 10.0（脊柱 9 个 + 双侧肩胛带/腕 26 个） |

**接触**：349 个几何中仅 **19 个**参与碰撞（floor、pelvis、股骨/胫骨/跟骨/趾骨皮肤、sacrum、skull），
摩擦 `[1.0, 0.005, 0]`，`solref=[0.02, 1.0]`，`priority=0`。

**42 条 equality 约束**（模型内部耦合，会产生约束力）：膝关节 8 条/侧（`knee_angle_r` 耦合冗余平移/旋转 DOF，
4 阶多项式）、`shoulder_elv` 10 条/侧（肩胛带线性耦合）、`elv_angle`/`deviation`/`flexion` 各 1 条/侧。
这就是 README 所说"206 joints 约束到 85"的实现方式。

### 1.2 动作 → excitation → activation → 力 → 力矩 → 动力学

代码路径（实测行号见 `reports/dynamics_audit.json`）：

```
策略输出 a_policy ∈ [-1,1]^700
  └─ MuscleNormWrapper.action():  a_env = 1/(1+exp(-5(a_policy-0.5))) ∈ (0,1)^700
      └─ gymnasium MujocoEnv._step_mujoco_simulation (mujoco_env.py:141)
          self.data.ctrl[:] = ctrl
          mujoco.mj_step(model, data, nstep=10)
          mujoco.mj_rnePostConstraint(...)
            ├─ 激活动力学: d(act)/dt = mju_muscleDynamics(ctrl, act, dynprm)
            ├─ 肌肉力:     data.actuator_force = mju_muscleGain(len,vel,…)·act + mju_muscleBias(len,…)
            ├─ 广义力:     data.qfrc_actuator（= 肌肉力沿 tendon 力矩臂）
            └─ 接触 / 约束 / 被动 / 重力 → data.qpos, data.qvel 下一时刻
  └─ 观测 = concat(qpos, qvel, qacc, act, actuator_force/1000, actuator_length,
                   actuator_velocity, key_xpos(相对骨盆), qpos_ref, qpos_ref_future×5, key_xpos_ref)  → 3601 维
```

**闭环公式经实证锁定**（`scripts/probe_muscle_closure.py`，`reports/muscle_closure_probe.json`）：

- 主动力 F0 位于 **`gainprm[2]`**，被动力位于 **`biasprm[2]`**
  （`gainprm` 实际 10 元，前 9 个语义为 `range[0], range[1], force(F0), scale, lmin, lmax, vmax, fpmax, fvmax`；
  后 6 个与 MuJoCo 默认值 200 / 0.5 / 1.6 / 1.5 / 1.3 / 1.2 逐一吻合，为交叉证据）。
- **同状态一致性检验误差 = 0.0**（700 块肌肉逐元素，其中 508 块有非零速度）：
  `data.actuator_force == mju_muscleGain(len,vel,lengthrange,acc0,gainprm[:9])·act + mju_muscleBias(len,lengthrange,acc0,biasprm[:9])`
- 运动轨迹上直接比较会出现最大 ~19 N 的残差，原因是 `actuator_velocity` 是**步后**采样，而 `mj_step`
  用步前速度算力再积分（半隐式 Euler）。这是采样口径问题，不是模型不一致——已用同状态检验排除。

### 1.3 checkpoint 的算法、版本与接口

| 项 | 值 |
|---|---|
| 算法 | `SAC_DynSyn`（SAC + DynSyn 层），138 个"肌群"动作 |
| 训练时版本 | SB3 **2.7.1**、PyTorch 2.10、Gymnasium 1.2.3、NumPy 2.4.2、Python 3.12 |
| 动作空间 | 700（策略内部先输出 138 维，`DynSynLayer.repeat_replace_x` 按组展开到 700，再 clamp 到 [-1,1]） |
| 观测空间 | 3601 |
| 观测归一化 | `VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10.0)`，统计量在 `checkpoint/best_env.zip`（裸 pickle） |
| 环境配置 | `skip_frames=10, reset_noise_scale=0.001, qpos_diff_th=0.06, gait_cycles=3, random_init=True`，`w_pelvis=100` |
| 训练步数 | 45,000,000（best model） |

**`dynsyn_weight_amp` 的评估语义（重要）**：checkpoint 的 `data` 中 `dynsyn_weight_amp = null`，
而 `DynSynLayer.forward` 在 `dynsyn_weight_amp is None` 时把 weight 置为 1，**不做协同权重重标定**，
即评估时动作 = 138 维肌群动作按组展开。上游 `eval.py` 走的正是这条路径。本工作区按此语义实现
（`hemirl/policy.py: build_group_expansion`），并用两条独立加载路径互验。

### 1.4 正常仿真 vs 运动学回放

`run_kinematic_play` 分支（`locomotionFull_v1.py:205`）会把 `data.qpos[:] = self.qpos_ref`、
`qvel=0`、`qacc=0`、`action*0`，并把重力置零——那是**运动学回放**。
`kinematic_play` 默认 `False`；`hemirl/envs.py: build_env()` **拒绝** `kinematic_play=True`，
并在构造后断言 `raw.unwrapped.kinematic_play is False`。

### 1.5 参考运动的使用方式

| 用途 | 位置 | 说明 |
|---|---|---|
| 观测 | `_get_obs`: `qpos_ref`、`qpos_ref_future`(5 步)、`key_xpos_ref` | 作为条件输入 |
| 奖励 | `_get_qpos_reward` / `_get_xpos_reward` / `_get_pelvis_reward` / `_get_healthy_reward` | 跟踪误差 |
| 初始化 | `reset_model` → `query_batch` → `set_state(init_qpos, init_qvel)` | **根节点只在 reset 时被参考设置** |
| 终止 | `is_healthy = mean|qpos[3:]−qpos_ref[3:]| ≤ qpos_diff_th` | 官方规则 |

### 1.6 是否存在"隐藏外力"

| 检查 | 结果 |
|---|---|
| `step()` 中对真实状态的赋值 | **仅 3 处**（`locomotionFull_v1.py:206-208`），且都在 `if self.kinematic_play:`（第 205 行）分支内 |
| `self.qpos_ref = …`（第 233 行） | 写的是**参考缓冲区**，不是仿真状态 |
| 根节点初始化 | 只在 `reset_model`（第 249 行）通过 `set_state()` |
| 环境源码中 `qfrc_applied` / `xfrc_applied` / `kp` / `kd` / `position_actuator` | **全部不存在** |
| 运行时 `‖qfrc_applied‖` / `‖xfrc_applied‖` 上限（80 步 × 2 种条件） | **0.0 / 0.0** |
| 额外的 pelvis 支撑力或外部稳定力 | **无**（`xfrc_applied` 恒为 0；`option.disableflags = 0` 说明重力/被动/接触/约束全部生效） |

---

## 2. 复现官方正常行走

```bash
cd /home/zrq/Documents/muscle-rl
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 1 --termination official --tag eval_min_official
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 5 --termination research --save-traj --tag eval_repeats_research
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 1 --termination official --video --tag eval_video
```

**结果**（`runs/eval_repeats_research/`，seed 0–4，确定性动作 `tanh(mean)`）：

| 指标 | 均值 ± 标准差 | 参考值 |
|---|---|---|
| 完成步数 / 时长 | **175 步 / 3.50 s**（5/5 全部走完，未触发终止） | 3 个步态周期 = 3.51 s |
| 水平位移 | 3.358 ± 0.046 m | — |
| 平均速度 | **0.960 ± 0.013 m/s** | 1.038 m/s（参考轨迹） |
| 骨盆高度（均值 / 最低） | 0.913 / 0.875 ± 0.003 m | 0.9205 / 0.9011 m |
| 关节跟踪误差（均值） | 0.0358 ± 0.0006 rad | 官方阈值 0.06 |
| 最大直立偏差 | 11.7° | 直立为 0° |
| 终止原因 | 5/5「未终止」 | — |

**视频**：`runs/eval_video/rollout.mp4`（175 帧 @ 50 fps，MUJOCO_GL=egl 离屏渲染），
关键帧见 `reports/video_frames/frame_090.png`——全身肌骨模型（骨架 + 肌肉）在真实动力学下迈步。

**结论**：官方全身肌骨模型 + 官方 DynSyn-SAC 策略在**真实动力学**（无运动学回放、无外力）下完成
3 个步态周期行走，速度达参考值的 92%，官方终止规则始终未触发。

---

## 3. 患侧肌力参数化

代码：`hemirl/muscle_actuator.py`（缩放）、`hemirl/muscle_groups.py`（分组映射）。

### 3.1 接口

```python
StrengthSpec(
    paretic_side='L'|'R',        # 患侧
    upper_scale=1.0,             # 患侧上肢 F0 倍率
    lower_scale=1.0,             # 患侧下肢 F0 倍率（与上肢独立）
    torso_scale=1.0,             # 躯干；默认 1.0 保持基准
    group_scales={...},          # 可选：整组键（'L/upper','R/lower','M/torso'）或具体肌肉名覆盖
    include_shoulder_girdle=False,# 是否把 44 块"权威归躯干、仅跨上肢关节"的肌肉计入上肢
    mode='active_only'|'active_and_passive',
)
```

语义：**F0_i = α_g · F0_i_baseline**，`g` 为肌肉所属组。
`StrengthScaler` 在构造时冻结不可变基准（`actuator_gainprm` / `actuator_biasprm` 的完整拷贝），
**每次 apply 都写成 `baseline × α`，绝不做 `*=` 累乘**。

### 3.2 主动产力 vs 被动产力（可区分模式）

| 模式 | 修改的字段 | 含义 |
|---|---|---|
| `active_only`（默认） | 仅 `gainprm[2]` | 肌肉**主动产力能力下降**，并联被动力不变 |
| `active_and_passive` | `gainprm[2]` 与 `biasprm[2]` | 主动 + 被动力共同缩放 |

**不修改**任何长度/时间参数（`lengthrange`、`scale`、`lmin/lmax`、`vmax`、`fpmax`、`fvmax`、`timeconst`），
也不动关节刚度、阻尼或策略动作幅值——那些都不是最大等长力。

### 3.3 肌肉分组映射（`reports/muscle_group_map.csv`）

两条独立证据链：
- **权威分组**（决定缩放范围）＝ 官方源文件 `Muscle/Muscle_{Leg,Arm,Arm_Hand,Torso}_{r,l}.xml`；
- **几何/功能验证**＝ 沿 tendon 的 wrap 对象找跨越的 rigid body，在运动学树上求「最低公共祖先 → 各 body」路径上的关节集合。

| 组键 | 数量 | 说明 |
|---|---|---|
| `R/lower` / `L/lower` | 50 / 50 | 髋、膝、踝、趾（与 `Muscle_Leg_r/l.xml` 计数一致） |
| `R/upper` / `L/upper` | 61 / 61 | 肩、肘、腕、手（与 `Muscle_Arm_r/l.xml` 计数一致） |
| `R/torso` / `L/torso` / `M/torso` | 208 / 239 / 31 | 含脊柱、肋骨、肩胛带肌（合计 478，与 `Muscle_Torso.xml` 一致） |
| 合计 | **700** | 组间无重叠、并集覆盖全部执行器 |

**两种边界情况（已在 CSV 中逐条标注，不隐藏）**：
1. **192 块肌肉不跨越任何可动关节**（多裂肌 `MF_*`、腹斜肌 `IO*`、骨间肌 `ExtIC_*/IntIC_*` 等）。
   原因：该简化模型脊柱只保留 `L5_S1`/`T12_L1`/`T1_head_neck` 三组活动关节，腰椎体与胸椎体之间是刚性连接。
   这是模型的真实属性——这类肌肉的缩放对关节力矩**没有影响**，不是错误。
2. **44 块权威归 torso、但只跨越上肢关节**的肌肉（斜方肌 `trap_*`、肩胛提肌 `levator_scap`、
   胸锁乳突肌 `cleid_*`、前锯肌 `SerrAnt*`）。默认不计入上肢组；
   实验需要时用 `include_shoulder_girdle=True` 显式纳入（R 22 块 / L 22 块）。

> 另需注意：官方 `Muscle_Arm_*.xml` 里包含了跨躯干的**背阔肌** `LD_*`（跨上肢 + 脊柱关节），
> 因此 61 块/侧的"上肢组"定义由官方文件决定，而不是纯几何判据。这一点在 CSV 的
> `crossed_limbs` 列可以直接核查。

---

## 4. 验证结果（任务书第七节）

### 4.1 单元测试 18/18（`reports/unit_tests.json`）

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_tests.py --with-heavy
```

```
18/18 通过                      ← tests/test_core.py
16/16 通过  ->  reports/verify_strength.json   ← scripts/verify_strength.py
```

### 4.2 逐项验证对照（`reports/verify_strength.json`）

| 要求 | 检查 | 结果 |
|---|---|---|
| — | F0 槽位实证（扰动-响应） | ✅ 主动 = `gainprm[2]`（比值 1.0999），被动 = `biasprm[2]`（比值精确 1.1） |
| 倍率 1.0 恢复基准 | `apply(1.0)` 后与基准**逐元素相等** | ✅ `max_abs_diff = 0.0` |
| 连续 0.5 再 0.75 == 基准×0.75 | 111 块肌肉倍率全为 0.75 | ✅ 与 0.75 的最大偏差 `1.1e-16`（浮点表示误差） |
| 左右侧、上下肢映射正确 | 恰为 `R/upper` 61 块 / `R/lower` 50 块；两组无交集；患对侧 0 块被改 | ✅ |
| 指定组外参数不变 | 589 块（健侧 + 躯干）逐元素不变 | ✅ `gain_equal/bias_equal = True` |
| 同姿态/速度/激活下主动力按倍率变化 | 上肢 0.5 → 力比 0.5001（min 0.5, max 0.504）；下肢 1.0 → 完全不变 | ✅ |
| 同上，多姿态重复 | 0.25 倍率下 3 个随机姿态力比 = 0.25 / 0.2501 / 0.25 | ✅ |
| 只改幅值不改 FL 曲线形状 | 4 个长度幅值下缩放比恒为 0.5001 | ✅ |
| 两种模式对被动力的影响可区分 | `active_only` 被动力逐元素不变；`active_and_passive` 按 0.5 缩放 | ✅ |
| 正常 step 无根节点参考覆盖 | 见 §5.1 / §5.3 | ✅ |

### 4.3 动力学审计（`reports/dynamics_audit.json`）

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/audit_dynamics.py --steps 80
```

**A. 静态代码审计**

- `kinematic_play` 默认 `False`
- `step()`（第 201–249 行）中对真实状态的赋值仅 3 处，全部位于 `if self.kinematic_play:`（第 205 行）之内
- `reset_model`（第 249 行）调用 `set_state()`——根节点只在 reset 设置
- 环境源码中不存在 `qfrc_applied` / `xfrc_applied` / `kp` / `kd` / `position_actuator`
- `gymnasium MujocoEnv._step_mujoco_simulation`（`mujoco_env.py:141`）=
  `data.ctrl[:] = ctrl` → `mj_step(model, data, nstep=10)` → `mj_rnePostConstraint`

**B. 模型结构审计**：700/700 为 muscle dyntype（`n_motor_actuators = 0`）、无 freejoint、
42 条 equality 全为关节耦合、无传感器、`disableflags = 0`（无禁用物理项）。

**C. 运行时审计（策略驱动 80 步）**

| 项 | 值 |
|---|---|
| `max‖qfrc_applied‖` / `max‖xfrc_applied‖` | **0.0 / 0.0** |
| 力项量级（均值范数） | 肌肉 1857 / 约束 1923 / 重力 617 / 被动 15.5 |
| `act` 均值 vs `ctrl` 均值 | 0.164 vs 0.139 → **不相等**，确认激活动力学在起作用 |
| 同状态肌肉力公式误差 | **0.0（exact = True，508 个执行器有非零速度）** |
| 与参考的偏差随时间增长 | 关节 0.0080 → 0.0383 rad；根节点位置 0.023 → 0.094 m → **状态未被参考覆盖** |
| 接触点数 | 均值 4.0（仅 19/349 几何参与碰撞） |

**C2. 把策略动作置零**（经 MuscleNormWrapper 后 excitation ≈ 0.076）：
骨盆高度 0.906 → **0.145 m**（重力主导下落）→ 证明运动由动力学产生，而非参考回放。

### 4.4 闭环链路证据小结

| 环节 | 证据 |
|---|---|
| 策略输出 → excitation | `MuscleNormWrapper` 与参考 sigmoid 逐点一致（21 点，误差 0）；动作均值 0.576–0.921（分组） |
| excitation → activation | `ctrl_mean ≠ act_mean`（0.139 vs 0.164），`act` 由 `mju_muscleDynamics` 以 `timeconst=[0.01,0.04]` 演化 |
| activation → 肌肉力 | 同状态公式误差 **0.0**（700 块，含 508 块非零速度） |
| 肌肉力 → 广义力矩 | `qfrc_actuator` 范数均值 1857，与约束力 1923 同量级并共同决定运动 |
| 广义力矩 → MuJoCo 动力学 | 接触（均值 4 点）、约束（42 条 equality）、被动（阻尼/armature/刚度）、重力齐备；无任何外力 |
| 下一状态 → 观测 | obs 3601 维含 `qacc`、`actuator_force`、`act`、`actuator_length/velocity`，闭环反馈成立 |

---

## 5. 终止配置

### 5.1 官方规则（用于复现，原样保留）

```python
is_healthy = mean(|qpos[3:] − qpos_ref[3:]|) ≤ qpos_diff_th   # 官方默认 0.06（checkpoint 用 0.06）
terminated = (not is_healthy) or time ≥ terminate_time × cycles
truncated  = time ≥ terminate_time × cycles
```

即**偏离参考姿态即终止**。`--termination official` 走这条路径。

### 5.2 研究规则（新增，`hemirl/termination.py`，`configs/termination_research.json`）

**不再因偏离参考姿态而终止**；改用物理上明确的跌倒 / 数值异常 / 任务失败条件：

| 条件 | 坐标含义 | 默认阈值 | 参数来源 |
|---|---|---|---|
| 骨盆高度过低 | 世界系 z（`data.xpos[pelvis][2]`，z 向上） | **0.55 m** | 参考轨迹骨盆高度均值 0.9205 m 的约 60% |
| 骨盆直立偏差过大 | 以参考姿态标定的直立轴与世界 z 轴夹角 | **60°** | 直立为 0°；需标定，见下 |
| 根节点平移速度过大 | `‖data.qvel[0:3]‖`（数值爆炸兜底） | 10 m/s | 参考步速 1.038 m/s 的 ~10 倍 |
| 根节点角速度过大 | `‖data.qvel[3:6]‖` | 50 rad/s | 兜底 |
| 关节速度范数过大 | `‖data.qvel‖` | 200 | 兜底 |
| 数值异常 | `qpos/qvel/qacc/act` 出现 NaN/Inf | 开 | 积分器与模型 |
| 超时 | 仿真时间 | 官方 `terminate_time × 3 = 3.51 s` | 官方回合长度 |

**直立轴为何要标定**：pelvis 局部 z 轴指向世界 −y（水平），全零姿态下与世界 z 夹角 90°。
直接用 body z 轴会把任何姿态判成跌倒。因此用参考姿态标定 `u_local = R_refᵀ·[0,0,1]`
（`hemirl/termination.py: pelvis_upright_local_axis`），参考姿态下偏差为 0。

`allow_reference_deviation_termination=True` 会被构造函数**直接拒绝**，从代码层面防止预设。

---

## 6. 最小肌力扫描实验

```bash
# 扫描 A（下肢=1.0，上肢 ∈ {1.0,0.75,0.5,0.25}）+ 扫描 B（上肢=1.0，下肢 ∈ 同）
MUJOCO_GL=egl PYTHONPATH=. python scripts/strength_sweep.py \
    --paretic-side R --seeds 0 1 2 3 4 --termination research --tag sweep_R_research
MUJOCO_GL=egl PYTHONPATH=. python scripts/strength_sweep.py \
    --paretic-side L --seeds 0 1 2 --termination research --tag sweep_L_research
```

**配对种子**：同一 seed 下所有倍率使用完全相同的初始条件（`env.reset(seed=...)` 决定轨迹时间、
初始噪声），策略权重固定，因此差异只来自肌力缩放。
模式 `active_only`；倍率经 `AppliedStrength` 落盘（含每块肌肉的实际倍率、被缩放肌肉数、组计数、实际写入的字段槽位）。

### 6.1 扫描 A：患侧 **上肢** 倍率（下肢固定 1.0），患侧 = R，n=5

| 上肢倍率 | 完成步数 | 时长 | 位移 | 平均速度 | 最低骨盆 | 跟踪误差 | 激活均值 | 终止 |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 175.0 | 3.50 s | 3.358 m | 0.9595 m/s | 0.875 m | 0.0358 | 0.1646 | 0/5 |
| 0.75 | 175.0 | 3.50 s | 3.405 m | 0.9729 m/s | 0.873 m | 0.0377 | 0.1638 | 0/5 |
| 0.50 | 175.0 | 3.50 s | 3.369 m | 0.9627 m/s | 0.867 m | 0.0413 | 0.1620 | 0/5 |
| 0.25 | 175.0 | 3.50 s | 3.237 m | 0.9248 m/s | 0.867 m | 0.0482 | 0.1613 | 0/5 |

**如实报告：无拐杖行走时，上肢肌力变化对结果影响很小。** 降到 25% 时速度仅下降 3.6%
（0.9595 → 0.9248），仍在种子噪声量级附近；跟踪误差上升（0.0358 → 0.0482）是更清晰的信号。
这与"行走主要由下肢驱动"一致（实测下肢肌肉力均值 ~94–127 N，上肢仅 ~14–18 N）。

患侧 = L（n=3）呈同向趋势：0.9591 → 0.9286 → 0.9155 → 0.8728 m/s，全部 175 步。

### 6.2 扫描 B：患侧 **下肢** 倍率（上肢固定 1.0），患侧 = R，n=5

| 下肢倍率 | 完成步数 | 时长 | 位移 | 最低骨盆 | 跟踪误差 | 激活均值 | 终止 |
|---|---|---|---|---|---|---|---|
| 1.00 | 175.0 | 3.50 s | 3.358 m | 0.875 m | 0.0358 | 0.1646 | 0/5 |
| 0.75 | 175.0 | 3.50 s | 3.235 m | 0.868 m | 0.0390 | 0.1720 | 0/5 |
| **0.50** | **90.0** | **1.80 s** | 1.944 m | 0.549 m | 0.0879 | **0.2407** | **5/5（跌倒）** |
| **0.25** | **37.0** | **0.74 s** | 0.716 m | 0.542 m | 0.0946 | 0.1929 | **5/5（跌倒）** |

终止原因：0.5 时 4/5 触发「骨盆高度 < 0.55 m」、1/5 触发「直立偏差 61.7° > 60°」；0.25 时 5/5 高度触发。

**可读出的现象**：
- 该策略能容忍 **25% 的患侧下肢肌力损失**（0.75 仍走完），但在 **50% 损失**下 5/5 跌倒，
  平均仅 1.80 s；25% 时 0.74 s 内即跌倒。
- 0.5 档位的**整体激活均值从 0.165 升到 0.241、肌肉力均值从 31.3 升到 44.2 N** ——策略在加大激活试图补偿，
  但最终失败；这与"固定权重策略未适应损伤"的预期一致。
- 患侧 = L（n=3）呈同样模式：1.00 与 0.75 走完 175 步（3.50 s）；0.50 → 平均 131 步 / 2.63 s，
  3/3 跌倒；0.25 → 平均 52 步 / 1.04 s，3/3 跌倒。基线速度两侧几乎相同
  （R 0.9595 / L 0.9591 m/s），侧别映射的对称性由此得到交叉验证。

> **解释边界**：以上只是**固定官方权重**下策略对肌力变化的响应，**不能**解释成"已经适应损伤的患者策略"。
> 也不涉及任何拐杖辅助——临界点假设此刻无法检验。

---

## 7. 交付物

| 交付项 | 位置 |
|---|---|
| 代码 | `hemirl/`（10 个模块）、`scripts/`（13 个入口）、`tests/test_core.py` |
| 配置 | `configs/termination_research.json` |
| 肌肉分组映射 | `reports/muscle_group_map.csv`（700 行，含跨越关节/body/site 与一致性标注）、`reports/muscle_group_summary.json` |
| 模型内省 | `reports/model_inspection_full.json`、`reports/root_layout.json` |
| 正常行走评估 | `runs/eval_min_official/`、`runs/eval_repeats_research/`（含 `run.json`/`episodes.json`/`summary.json`/轨迹 npz） |
| 视频 | `runs/eval_video/rollout.mp4` + `reports/video_frames/frame_090.png` |
| 动力学审计 | `reports/dynamics_audit.json`、`reports/muscle_closure_probe.json` |
| 肌力验证 | `reports/verify_strength.json`、`reports/unit_tests.json` |
| 扫描实验 | `runs/sweep_R_research/`、`runs/sweep_L_research/`（各含 `sweep.csv`/`episodes.json`/`summary.json`/`run.json`） |
| 依赖清单 | `requirements-hemirl.txt` |
| 工程约定 | `AGENTS.md`、`README.md`、`.gitignore` |

启动命令见 `README.md`「快速开始」。

---

## 8. 未解决问题与阻塞

### 8.1 已解决（但需要你知道）

1. **`env_isaaclab` 被我误伤并已修复**。
   `pip install stable_baselines3` 因 pip 配置了 `extra-index-url=https://pypi.nvidia.com`
   而把 **torch 2.7.0+cu128 升级到 2.14.0**，破坏了 isaacsim/torchvision/torchaudio。
   已还原：卸载 SB3、`pip install torch==2.7.0+cu128 triton==3.3.0`、
   卸载 cu13 系列包，并**强制重装 `nvidia-cudnn-cu12==9.7.1.26` 与 `nvidia-nccl-cu12==2.26.2`**
   （卸载 cu13 包时连删了同路径的 cu12 库文件）。
   现状：`torch 2.7.0+cu128` 可 import、`torch.optim.Adam` 可用、CUDA 可用、SB3 已清除。
   之后本项目改用独立环境 `hemirl`，不再触碰 `env_isaaclab`。
2. **`torchaudio` 的 CUDA 版本不匹配是既有状态**（torchaudio 2.7.0 是 cu126 构建，torch 是 cu128），
   与本次改动无关，也不是我引入的。若你的 Isaac Lab 流程需要 torchaudio，需要单独对齐。
3. **checkpoint 的 `lr_schedule` 是 Python 3.12 的 code object**。在 Python 3.11 环境下
   `COMPARE_OP` 编码不兼容，任何输入都会走错分支并对 `warmup_fraction=0` 做除法
   （实测 `fn(1.0)` 直接 `ZeroDivisionError`），导致 SB3 建优化器时 `predict` 前就崩。
   解决：主环境改用 Python 3.12；同时保留一个仅在加载期生效的回退
   （`hemirl/policy.py: patched_float_schedule`），加载完立即恢复。评估不更新参数，学习率不影响结果。
4. **numpy 2.x 的 pickle 在 numpy 1.x 下不可读**（`numpy._core`、随机数构造器签名、`PCG64.__setstate__`
   的 `(state, seed_sequence)` 形式）。主环境用 numpy 2.3.5 无此问题；
   仍保留 `hemirl/numpy_compat.py` 作为防御（在 numpy ≥ 2 时为空操作）。
5. **无显示环境的渲染**：`MUJOCO_GL=egl` 可用；`osmesa` 不可用（系统缺 `libOSMesa`，
   PyOpenGL 报 `'NoneType' object has no attribute 'glGetError'`）。视频已成功导出。

### 8.2 仍未解决 / 需要注意

1. **`dynsyn_weight_amp=null` 的语义歧义（上游问题）**。
   训练末段该值为 0.1（`min(k(t−a), 0.1)`），但 `null` 在加载后使 `DynSynLayer` 退化为"权重组=1"。
   上游 `eval.py` 走的就是退化路径，本工作区据此复现。**如果**你希望评估的是训练末段的 0.1 权重，
   需要显式设置，这是另一个（也可能更接近论文数字的）口径——目前未启用，结果里已说明。
2. **回合长度受官方参考轨迹限制（3.51 s / 3 个步态周期）**。
   平衡研究的临界点假设需要更长时程；`trajectory.query_batch` 在超出轨迹时长后的行为需要先核实，
   才能安全地延长回合（见 §9 建议 3）。
3. **研究终止阈值是启发式的**（0.55 m / 60°），量级由参考轨迹统计标定，不是文献值。
   若要发表，需要用跌倒定义做敏感性分析。
4. **上肢肌力对无拐杖行走影响很小**（见 §6.1）。这是真实结果，不是 bug；
   但它意味着**单靠上肢肌力扫描无法检验临界点假设**——必须引入拐杖/支撑交互。
5. **`qfrc_actuator` 与 `qfrc_constraint` 量级相当**（1857 vs 1923）。
   42 条 equality 约束（膝、肩胛带）承担了相当部分广义力，后续如果要解释"关节力矩"，
   需要把约束力与肌肉主动力分开陈述（本工作区已在审计中分别记录）。
6. **contact force 依赖 `mj_contactForce`**（模型无 contact sensor，`nsensor=0`）。
   指标可用，但只有 19 个几何参与碰撞，无法得到逐足多区域压力分布。

---

## 9. 下一阶段建议

### 9.1 从"正常策略"走向"偏瘫适应训练"

1. **不要改动官方 checkpoint 的动作/观测接口**（`AGENTS.md` 硬性规则 5）。
   新增观测（如患侧特定的激活上限、疲劳状态、辅助接触力）一律放进**独立配置 + 重新训练**。
2. **损伤参数化先于训练定稿**：把 `StrengthSpec` 作为训练时的域随机化维度
   （上肢/下肢倍率各自在 [0.25, 1.0] 采样），让策略学会在多档肌力下维持平衡；
   再固定倍率评估，才能把"适应后策略"与"固定策略"区分开。
3. **奖励需要重设计**：当前奖励是参考轨迹跟踪（`w_qpos/w_xpos/w_pelvis`），
   与"平衡能力"并不等价——跟踪得好可能只是因为参考动作与损伤后的动力学相容。
   建议新增一个**不依赖参考轨迹**的平衡任务（维持站立/走位、惩罚跌倒与过度躯干摆动），
   或者把跟踪奖励降权并加入质心/支撑多边形项。这属于新配置，不覆盖官方环境。
4. **时程需要延长**：`gait_cycles` 只到 3。需要先核实 `imitation_trajectory.query_batch`
   在超出轨迹时长后的行为（循环 or 钳制），再决定是循环参考还是改为自由行走任务。
5. **被动模式对照**：把 `active_only` 与 `active_and_passive` 作为消融；
   如果结论对模式敏感，说明"主动产力下降"与"主动+被动共同下降"在机制上确实不同。

### 9.2 交付物 5 的补充：加入双侧可切换拐杖需要哪些接口

要支持"无辅助 / 健侧拄拐 / 患侧拄拐"三条件切换，至少需要以下接口（目前**都还没有**，
当前环境是纯无辅助行走）：

| 需要的能力 | 现状 | 需要新增 |
|---|---|---|
| 拐杖刚体与几何 | 无 | 派生 XML（写到 `artifacts/`，不改 `external/`）：杖身 capsule + 底部接触球，挂在 `hand_r`/`hand_l` 上 |
| 手–拐杖耦合 | 无 | 固定 weld（`mjEQ_WELD`）或 `<equality><connect>`；需要**双侧可切换** → 用 `eq_active` 数组在运行时开关，或两套派生模型 |
| 拐杖执行器/被动 | 无 | 若让策略控制拐杖，需要额外的 actuator（会改变 `nu`，**必须**重新训练）；若只是被动支撑，则不加 actuator，仅靠接触 |
| 地面接触 | 已有 floor + 19 个接触几何 | 增加拐杖底部与 floor 的 contype/conaffinity；需要为拐杖设置合适摩擦（`[1.0, 0.005, 0]` 与足部一致或更高） |
| 观测扩展 | 3601 维固定 | 新增拐杖尖端位置/速度、拐杖-地面接触力、哪一侧在用（one-hot）；放进**新配置**，不动官方接口 |
| 奖励项 | 现有 4 项 | 新增：拐杖载荷比例、躯干直立、质心-支撑多边形位置、患侧足摆动对称性；**不要**预设"患侧辅助更好" |
| 实验编排 | `StrengthSpec` 已有 `paretic_side` + 上下肢独立倍率 | 新增 `AssistCondition(assist_side=None|'L'|'R')`，与 `StrengthSpec` 正交组合；扫描时用配对种子共享初始条件 |
| 结果指标 | 已有速度/位移/骨盆高度/直立偏差/接触计数 | 新增：拐杖载荷时序、支撑多边形、双侧步长/摆动时间对称性、稳定裕度 |

**最省事的可行路径**：先做**被动拐杖**（不加 actuator，只加刚体+接触+`eq_active` 开关），
这样可以继续用官方 checkpoint 评估"支撑改变了的动力学"；
等到要做"用拐杖主动发力"时再重新训练（`nu` 会变）。

---

## 10. 一句话结论

第一阶段目标全部达成：官方全身肌骨模型 + 官方 DynSyn-SAC 策略在**真实动力学**下复现了 3 个步态周期的
正常行走（0.96 m/s，全程无外力、无运动学回放，公式闭环误差为 0）；
肌力参数化从不可变基准出发、上下肢与左右侧独立可配、两种主动/被动模式可区分，
并通过 18 项单元测试与 16 项专项验证；扫描实验显示**固定权重策略可容忍 25% 的下肢肌力损失但在 50% 时跌倒**，
而**上肢肌力在无辅助行走中影响很小**——后者如实报告，也说明临界点假设必须引入拐杖交互才能检验。
