# AGENTS.md — muscle-rl 工作区约定

本工作区是 `muscle-pt` 项目（`/home/zrq/Documents/muscle`）的新增**独立后端**，用于
MS-Human-700 全身肌骨模型 + 强化学习（msgym）的偏瘫研究。**不覆盖**原项目的 BIO 模型、
患者 BVH 处理与分析功能。原项目路径：`/home/zrq/Documents/muscle`（git remote `myrepo`）。

## 目录约定

| 路径 | 作用 |
|---|---|
| `external/` | 上游依赖源码（只读，勿改；用 `scripts/setup_external.sh` 按 commit 获取，**不入 git**） |
| `external/MS-Human-700/` | 官方肌骨模型 XML + 资产（独立克隆，pin 到具体 commit） |
| `external/msgym/` | 官方 Gymnasium 环境 + DynSyn-SAC 训练/评估脚本 |
| `hemirl/` | **本工作区新增代码**：肌力参数化、终止配置、评估与扫描、审计与验证 |
| `configs/` | 实验配置（JSON） |
| `scripts/` | 可复制运行的 CLI 入口（薄封装，逻辑在 `hemirl/`） |
| `artifacts/` | 下载的 checkpoint、派生模型（不入 git） |
| `runs/` | 每次运行一个子目录；`run.json`/`episodes.json`/`summary.json`/`sweep.csv` **入库**，轨迹 `*.npz` 与 `*.mp4` 不入库 |
| `reports/` | 阶段报告、映射清单、验证输出、视频关键帧 |

## 硬性规则

1. **不修改 `external/` 下的文件**（上游只读）。需要派生模型时在运行时改编译后的 `MjModel`
   字段，或把派生 XML 写到 `artifacts/`。
2. **不用运动学回放代替动力学**。`kinematic_play=True` 仅允许用于数据/接口自检，且必须在
   结果里显式标注。
3. **肌力缩放必须从不可变基准计算**，禁止累乘（见 `hemirl/strength.py`）。
4. **不得在奖励/数据处理/结果展示里预设患侧辅助最终更好**。临界点是待检验假设。
5. 新增观测只能放在后续重新训练用的独立配置里，**不得**改变官方 checkpoint 的观测/动作接口。
6. 每次运行必须落盘 provenance：仓库 commit、checkpoint 来源与哈希、依赖版本、随机种子。
7. **上游源码与大体积产物不入 git**（`external/`、`artifacts/`、`*.npz`、`*.mp4`）。
   上游用 `scripts/setup_external.sh` 按 commit 复原，checkpoint 按 `README.md` 的 Release 链接下载。

## 环境

- **本项目专用环境：conda `hemirl`**（`/home/zrq/miniconda3/envs/hemirl`）
  - Python **3.12**：与 checkpoint 的序列化环境一致（官方 checkpoint 用 Python 3.12 保存，
    其 cloudpickle 代码对象在 3.11 上字节码不兼容）
  - numpy **2.x**、mujoco 3.11.0、gymnasium、torch（CPU 版即可，推理不需要 GPU）
  - `stable_baselines3==2.7.1`、`sb3_contrib==2.7.1`（与 checkpoint 的训练版本一致）
  - 安装依赖时务必用 `PIP_CONFIG_FILE=/dev/null`，否则会走 `pypi.nvidia.com` 并升级 torch
- **禁止**向 `env_isaaclab` 安装任何本项目依赖：该环境服务于 `muscle-pt`（Isaac Lab），
  pip 会把 torch 2.7.0 升级到 2.14.0 从而破坏 isaacsim/torchvision/torchaudio。
  （2026-09-16 发生过一次，已还原；详见 `/memories/repo/environment.md`）
- 运行命令统一形式：`PYTHONPATH=. python scripts/<entry>.py ...`
- 无显示环境渲染：`MUJOCO_GL=egl` 可用；`osmesa` 不可用（系统未安装 `libOSMesa`）。
  渲染失败时保存轨迹 npz 并说明原因。

## 术语

- **患侧 (paretic)**：`L` 或 `R`，与 XML 中 `_l` / `_r` 后缀一致。
- **F0**：肌肉最大等长力（MuJoCo muscle actuator 的 `gainprm[0]`，对应 XML `force`）。
- **upper / lower**：上肢（肩/肘/腕/手）与下肢（髋/膝/踝/趾）肌群，**独立缩放**。
