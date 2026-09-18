# muscle-rl

`muscle-pt` 项目的**独立新增后端**：MS-Human-700 全身肌骨模型 + msgym 强化学习环境，
用于偏瘫（hemiplegia）上肢肌力研究的第一阶段工作。

- **不覆盖**原项目（`/home/zrq/Documents/muscle`）的 BIO 模型 / 患者 BVH 处理与分析功能。
- 上游依赖只读放在 `external/`，本项目新增代码在 `hemirl/`。
- **技术路线与设计理由见 [`docs/technical_route.md`](docs/technical_route.md)。**

## 第一阶段已完成

1. **复现官方全身肌骨模型预训练 RL 行走**：官方 DynSyn-SAC checkpoint（全身模型 700 肌肉）
   在真实动力学下走完 3 个步态周期（**176 步 / 3.52 s**），平均速度 **0.958 ± 0.013 m/s**，
   官方终止规则未触发（5/5 均为时间上限，非参考偏差）；已导出视频
   （`*.mp4` 不入库，关键帧见 `reports/video_frames/`）。
2. **可重复、可恢复的患侧肌力参数化**：`hemirl/muscle_actuator.py`，
   从不可变基准计算 F0 倍率，支持左右侧 / 上下肢独立缩放、两种主动-被动模式。
3. **动力学闭环核查**：`hemirl/dynamics_audit.py` + `hemirl/forces.py`，
   静态代码审计 + 模型结构审计 + 运行时审计 + 约束力分类分解，证据见 `reports/dynamics_audit.json`。
4. **统一的研究环境**：`hemirl/research_env.py`（训练与评估共用），
   严格区分 `terminated`（物理跌倒 / 数值异常）与 `truncated`（规定时长 / 本地上限），
   记录**每一步（含终止步）**，运行时间取自**实际仿真时间差**。
5. **策略一致性验证**：`hemirl/policy_verify.py`，官方 SB3 路径与纯 torch 直读路径
   在 126 个观测上逐元素一致（`max|Δa| = 5.96e-08 ≤ 1e-5`）。
6. **环境接口与实验记录**：每次运行落盘完整 provenance（HEAD + dirty 补丁哈希 /
   上游 commit 与脏文件分类 / checkpoint 与模型文件哈希 / 依赖版本 / 种子 / 实际肌力倍率）。

### 已知结论（如实记录）

* **官方策略无法完成 20 s 长时行走**：5 个种子全部在 4.46–4.90 s 因侧向失稳跌倒
  （骨盆直立偏差 > 60°）。归因分析表明**不是**参考轨迹循环造成（周期边界跳变小于周期内
  正常单步变化），而是策略自身的长时稳定性问题。详见
  [`reports/phase2_acceptance_report.md`](reports/phase2_acceptance_report.md) 第 4.2 节。
* **下肢损伤是主因**：下肢肌力 0.5 时基本全倒，0.25 时平均存活 < 1 s；
  上肢降到 0.25 仍能走完全程（速度仅 −3.6%）。
* 以上均为**固定官方策略**的评估结果，**不是**偏瘫适应训练结果。

## 第二阶段：正常肌力长时稳定性（已完成，未达门槛）

详见 [`reports/phase3_long_horizon_report.md`](reports/phase3_long_horizon_report.md)。

* **诊断**：侧向漂移是唯一在跌倒前显著发散的信号（人体 0.438 m vs 参考自身 0.076 m）；
  动作饱和从 1.42 s 起就相对基线变化；只靠「物理存活」奖励无法早期发现失稳。
  观测裁剪经**命名对照实验**证实是**结果而非原因**（取消裁剪：4.62 s → 4.61 s）。
* **微调入口**：`scripts/train_healthy.py`（含 `--resume`）/ `scripts/eval_healthy.py` /
  `configs/train_healthy_v1.json`。沿用 DynSyn-SAC 与官方 actor，显式固定
  `dynsyn_weight_amp = 0.0`（否则上游 `train()` 会算出 0.075 并静默改变动作语义）。
* **初始化定为 keep**：4 组共 2000-transition 对照显示，重置 critic 会把验证集存活
  从 4.95 s 打到 **0.64 s**。
* **试训练**：50,604 transitions / 12.51 分钟。配对评估（20 个测试种子，20 s）：
  存活 **4.97 → 5.60 s**、速度 **0.893 → 0.965 m/s**、最大侧偏 **0.450 → 0.294 m**、
  最大倾角 **61.1° → 29.4°**；但 20 s 完成率仍为 **0/20**。
* **结论**：**未达到**进入损伤适应训练的门槛（要求 18/20 完成 20 s）。
  下一次最有依据的单项调整：给侧向偏移加**终止条件**（而非继续用弱的密集惩罚），
  并把训练预算提高到与 critic 收敛需求匹配（当前 Q 高估 36%）。

## 环境

```bash
conda activate hemirl          # Python 3.12, mujoco 3.11, gymnasium 1.2.3, SB3 2.7.1, torch CPU
cd /home/zrq/Documents/muscle-rl
```

依赖清单见 `requirements-hemirl.txt`。**不要**向 `env_isaaclab` 安装本项目依赖（会破坏 Isaac Lab）。

## 首次准备（获取上游依赖与 checkpoint）

上游源码与大体积产物**不入本仓库**，按下面方式复原（均在仓库根目录执行）：

```bash
# 1) 获取上游只读依赖，并 pin 到与实验一致的 commit
bash scripts/setup_external.sh

# 2) 下载官方 checkpoint（约 214 MB）
mkdir -p artifacts/checkpoints && cd artifacts/checkpoints
curl -L -O https://github.com/LNSGroup/msgym/releases/download/Checkpoints/LocomotionFull.zip
unzip -q LocomotionFull.zip && cd ../..

# 3) 自检（18 项核心单元测试 + 11 项研究环境回归 + 22 项肌力专项）
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_tests.py --with-heavy
```

`setup_external.sh` 会把两个上游仓库 fetch 到指定 commit 并校验 HEAD
（`MS-Human-700 @ 2d68695`、`msgym @ ad3aac1`），同时校验模型的 `MS-Human-700.xml` sha256。

## 目录

| 路径 | 说明 |
|---|---|
| `external/MS-Human-700/` | 官方模型 XML + 资产（commit `2d68695`，用 `scripts/setup_external.sh` 获取，不入库） |
| `external/msgym/` | 官方 Gymnasium 环境与 DynSyn-SAC 脚本（commit `ad3aac1`，同上） |
| `hemirl/` | 本项目代码：肌群映射、肌力缩放、研究环境、终止配置、评估、审计、力分解、渲染 |
| `scripts/` | CLI 入口（可复制运行） |
| `configs/` | 实验配置 |
| `docs/` | 说明性文档（技术路线等） |
| `artifacts/checkpoints/LocomotionFull/` | 官方 checkpoint（GitHub Release） |
| `runs/` | 每次运行的 `run.json` / `episodes.json` / `summary.json` / `sweep.csv`（轨迹 `*.npz` 与视频不入库） |
| `reports/` | 模型内省、肌群映射、审计与验证输出、阶段报告 |

## 快速开始

```bash
# 0) 验收：一条命令跑完全部验证（失败/必需项被跳过时返回非零退出码）
MUJOCO_GL=egl PYTHONPATH=. python scripts/acceptance.py

# 1) 官方行走最小复现（官方终止规则，保留原语义）
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 1 --termination official --tag eval_min_official

# 2) 短时基线复验（官方配置，5 个种子）
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 5 --seed 0 --termination official --tag eval_short_baseline_official

# 3) 研究环境长时评估（20 s；先做参考连续性前置检查）
MUJOCO_GL=egl PYTHONPATH=. python scripts/probe_reference_continuity.py --cycles 20
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 5 --seed 0 --termination research --max-episode-seconds 20 \
    --save-traj --tag eval_long_research

# 4) 导出视频（含终止帧）
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 1 --termination research --max-episode-seconds 20 \
    --video --tag eval_long_research_video

# 5) 动力学闭环审计（含约束力分类分解）
MUJOCO_GL=egl PYTHONPATH=. python scripts/audit_dynamics.py --steps 80

# 6) 肌力扫描（A 上肢 / B 下肢，配对种子；走完与跌倒分开统计）
MUJOCO_GL=egl PYTHONPATH=. python scripts/strength_sweep.py \
    --paretic-side R --seeds 0 1 2 3 4 --termination research --tag sweep_R_short_research

# 7) 肌肉分组映射导出
PYTHONPATH=. python scripts/export_muscle_map.py
```

显式指定肌肉倍率（可附在 `eval_official.py` 上）：

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 3 --termination research \
    --paretic-side R --upper-scale 0.5 --lower-scale 1.0 \
    --strength-mode active_only --tag eval_R_upper0.5
```

### 两种评估模式的差别

| 模式 | `terminated` | `truncated` |
|---|---|---|
| `--termination official` | 上游 `not is_healthy`（偏离参考姿态）或达到官方 3.51 s | 达到官方 3.51 s |
| `--termination research` | 物理跌倒 / 数值异常（单独归类） | 达到 `--max-episode-seconds`（默认 20 s）或本地控制步上限 |

### 代码入口

| 用途 | 入口 |
|---|---|
| 官方环境（复现 checkpoint 原语义） | `hemirl/envs.py: build_env()` |
| **研究环境（训练与评估共用）** | `hemirl/research_env.py: build_research_env()` |

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/technical_route.md`](docs/technical_route.md) | **技术路线**：闭环数据流、三层技术选择、关键工程决策、与旧路线的差异、下一步 |
| [`reports/phase3_long_horizon_report.md`](reports/phase3_long_horizon_report.md) | **第二阶段报告**：长时失稳诊断、微调入口与资源开销、有限预算试训练、配对评估、下一次单项调整 |
| [`reports/phase2_acceptance_report.md`](reports/phase2_acceptance_report.md) | **第一阶段修复验收报告**：逐项状态分类、口径变更、长时评估失败归因、下一阶段入口与阻塞 |
| [`reports/phase1_report.md`](reports/phase1_report.md) | **阶段一历史记录**：代码核查结论、复现数据、验证结果（数字使用**修复前**的记录口径） |
| `reports/acceptance.json` | 验收命令的结果（passed / failed / skipped） |
| [`AGENTS.md`](AGENTS.md) | 工作区硬性约定（上游只读、不用运动学回放、禁止预设结论等） |
