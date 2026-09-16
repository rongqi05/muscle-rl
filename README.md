# muscle-rl

`muscle-pt` 项目的**独立新增后端**：MS-Human-700 全身肌骨模型 + msgym 强化学习环境，
用于偏瘫（hemiplegia）上肢肌力研究的第一阶段工作。

- **不覆盖**原项目（`/home/zrq/Documents/muscle`）的 BIO 模型 / 患者 BVH 处理与分析功能。
- 上游依赖只读放在 `external/`，本项目新增代码在 `hemirl/`。
- **技术路线与设计理由见 [`docs/technical_route.md`](docs/technical_route.md)。**

## 第一阶段已完成

1. **复现官方全身肌骨模型预训练 RL 行走**：官方 DynSyn-SAC checkpoint（全身模型 700 肌肉）
   在真实动力学下走完 3 个步态周期（175 步 / 3.50 s），平均速度 0.96 m/s，
   官方终止规则未触发；已导出视频（`*.mp4` 不入库，关键帧见 `reports/video_frames/`）。
2. **可重复、可恢复的患侧肌力参数化**：`hemirl/muscle_actuator.py`，
   从不可变基准计算 F0 倍率，支持左右侧 / 上下肢独立缩放、两种主动-被动模式。
3. **动力学闭环核查**：`hemirl/dynamics_audit.py` + `scripts/audit_dynamics.py`，
   静态代码审计 + 模型结构审计 + 运行时审计，证据见 `reports/dynamics_audit.json`。
4. **环境接口与实验记录**：`scripts/eval_official.py`、`scripts/strength_sweep.py`，
   每次运行落盘 provenance（commit / checkpoint 哈希 / 依赖版本 / 种子）。

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

# 3) 自检（18 项单元测试 + 16 项肌力专项验证）
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_tests.py --with-heavy
```

`setup_external.sh` 会把两个上游仓库 fetch 到指定 commit 并校验 HEAD
（`MS-Human-700 @ 2d68695`、`msgym @ ad3aac1`），同时校验模型的 `MS-Human-700.xml` sha256。

## 目录

| 路径 | 说明 |
|---|---|
| `external/MS-Human-700/` | 官方模型 XML + 资产（commit `2d68695`，用 `scripts/setup_external.sh` 获取，不入库） |
| `external/msgym/` | 官方 Gymnasium 环境与 DynSyn-SAC 脚本（commit `ad3aac1`，同上） |
| `hemirl/` | 本项目代码：肌群映射、肌力缩放、终止配置、评估、审计、渲染 |
| `scripts/` | CLI 入口（可复制运行） |
| `configs/` | 实验配置 |
| `docs/` | 说明性文档（技术路线等） |
| `artifacts/checkpoints/LocomotionFull/` | 官方 checkpoint（GitHub Release） |
| `runs/` | 每次运行的 `run.json` / `episodes.json` / `summary.json` / `sweep.csv`（轨迹 `*.npz` 与视频不入库） |
| `reports/` | 模型内省、肌群映射、审计与验证输出、阶段报告 |

## 快速开始

```bash
# 0) 单元测试 + 肌力验证（无需 checkpoint 之外的大依赖）
MUJOCO_GL=egl PYTHONPATH=. python scripts/run_tests.py --with-heavy

# 1) 官方行走最小复现（官方终止规则）
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 1 --termination official --tag eval_min_official

# 2) 多次重复（研究终止规则，保存轨迹）
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 5 --termination research --save-traj --tag eval_repeats_research

# 3) 导出视频
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 1 --termination official --video --tag eval_video

# 4) 动力学闭环审计
MUJOCO_GL=egl PYTHONPATH=. python scripts/audit_dynamics.py --steps 80

# 5) 肌力扫描（A 上肢 / B 下肢，配对种子）
MUJOCO_GL=egl PYTHONPATH=. python scripts/strength_sweep.py \
    --paretic-side R --seeds 0 1 2 3 4 --termination research --tag sweep_R_research

# 6) 肌肉分组映射导出
PYTHONPATH=. python scripts/export_muscle_map.py
```

显式指定肌肉倍率（可附在 `eval_official.py` 上）：

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
    --episodes 3 --termination research \
    --paretic-side R --upper-scale 0.5 --lower-scale 1.0 \
    --strength-mode active_only --tag eval_R_upper0.5
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/technical_route.md`](docs/technical_route.md) | **技术路线**：闭环数据流、三层技术选择、关键工程决策、与旧路线的差异、下一步 |
| [`reports/phase1_report.md`](reports/phase1_report.md) | **阶段报告**：代码核查结论、复现数据、验证结果、未解决问题 |
| [`AGENTS.md`](AGENTS.md) | 工作区硬性约定（上游只读、不用运动学回放、禁止预设结论等） |
