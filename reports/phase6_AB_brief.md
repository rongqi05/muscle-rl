# 阶段 A/B 简报：断点续训修复与物理失稳诊断（2026-10-19）

## A. 断点续训（已修复并验证）

**修复前**：`model.save()` 明确排除 replay buffer（SB3 `_excluded_save_params`）；
`load()` 只重建空 buffer；`CurriculumCallback` 新建时 `stage_idx=0` 且 worker 时长
停在 `stages[0]`。→ 旧 `--resume` 实际是「权重续训 + 空经验池 + 课程归零」。

**修复后**（commit `9970a28`）：

| 恢复项 | 机制 |
|---|---|
| replay buffer | `<tag>_replay_buffer.pkl` 单独保存（`save_replay_buffer`）；不存在时显式标记「v1 权重热启动、新 buffer」 |
| 课程状态 | `<tag>_curriculum_state.json`（stage_idx / 阶段内计数 / 事件历史 / buffer pos/full） |
| worker 时长 | `_on_training_start` 向全部 worker 广播当前阶段（`env_method`，不重建进程） |
| 模型/优化器 | SB3 `_get_torch_save_params` 含 `policy` + `actor/critic.optimizer` + `ent_coef_optimizer` → 权重与 Adam 状态都随 zip 恢复 |
| 归一化 | `<tag>_env.zip` 与模型同目录保存 |
| 语义固定 | 恢复后重设 `dynsyn_weight_amp=0.0` 与常数 lr |
| 内存 | 恢复只加载 `last`（不再走 `load_official_model`，避免官方 1e6 buffer 瞬时 29.4 GiB）；峰值 RSS 与耗时落盘 |

**实测**（`scripts/check_resume.py`，两段各 600 transitions，buffer=2000 以控制体积，
`reports/resume_check.json`）：**12/12 通过**。关键项：buffer 文件 63.0 MiB、恢复后
`pos` 与保存值一致（49）、第二段接续（49→98 行 = 每行 12 个 transition）；
**续训保真 max|Δa| = 0.000e+00**（加载+重设 lr/amp 后、训练前）；归一化逐位一致；
Adam 状态保留；`num_timesteps` 45,000,600 → 45,001,200、`_n_updates` 接续；
amp 仍为 0.0；峰值内存 1.50 GiB、恢复 0.6 s（小 buffer 口径）。
生产口径（buffer 1e5 = 2.94 GiB）的保存/恢复耗时与体积将在阶段 C 的周期检查点里实测记录。

测试：C1–C11（课程）11/11；C10/C11 为新增加。

## B. 物理失稳诊断（v1，开发种子 301–305，冻结归一化，确定性）

`scripts/diagnose_physics.py` → `reports/phase6_physics_diag.{json,png}`。
逐控制步记录：相位、左右足法向/竖直 GRF、总接触力、骨盆与质心高度/竖直速度、
侧向位置/速度、髋（屈/内收/旋转）/膝/踝肌肉驱动关节力矩（`qfrc_actuator`）、
关键肌群（glmax/glmed/vas/hams/soleus/gastroc/tibant）激活与力、奖励分项。

**领先量（5 回合中位，相对终止；双侧 3σ 越界，窗口 4 s）**：

| 时间 | 物理量 |
|---|---|
| −3.4 s | **胫前肌（tibant_r）力/激活** |
| −3.2 s | **股四头肌（vas_r）激活**、**侧向偏移**、**侧向速度** |
| −3.0 s | 髋屈/髋旋/膝力矩、臀大/腘绳肌激活、比目鱼肌 |
| −2.6 s | **骨盆/质心竖直速度转负** |
| −1.9 s | 踝力矩（左） |
| −0.76 s | **左右负荷失衡** |
| −0.5 s | **骨盆/质心高度开始掉** |
| −0.02 s | 总接触力崩掉（终止） |

**读法（时间顺序 ≠ 因果）**：最早可观察的变化是**支撑肌群（胫前/股四）的激活偏离**，
与**侧向漂移同时**（约 −3.2 s）；竖直速度在约 −1 s 之后才开始向下，负荷失衡与高度
下降是最后 0.8 s 的事。即：**高度塌陷是链的末端，不是起点**；接触变化是最晚的。

这一证据把阶段 D 的候选改动指向两条路（只选一条）：
1. **竖直支撑**：在竖直速度转负（−2.6 s）处给连续的支撑相关信号；
2. **侧向/落脚**：在侧向漂移（−3.2 s）处加强侧向速度/落脚控制。

（诊断是只读的；未做任何外加力/扰动，也不会进入训练与验收。）
