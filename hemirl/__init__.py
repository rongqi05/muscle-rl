"""hemirl — 偏瘫 (hemiplegia) 肌骨强化学习研究后端。

本包为 `muscle-pt` 项目的独立新增后端，基于：
- MS-Human-700 全身肌骨模型 (https://github.com/LNSGroup/MS-Human-700)
- msgym Gymnasium 环境 (https://github.com/LNSGroup/msgym)

第一阶段目标：复现官方预训练行走 + 可重复/可恢复的患侧肌力参数化 + 动力学闭环核查。
"""

from hemirl.paths import (  # noqa: F401
    WORKSPACE_ROOT,
    MSHUMAN_ROOT,
    MSGYM_ROOT,
    MODEL_XML,
    CHECKPOINT_ROOT,
    RUNS_ROOT,
    REPORTS_ROOT,
)

__all__ = [
    "WORKSPACE_ROOT",
    "MSHUMAN_ROOT",
    "MSGYM_ROOT",
    "MODEL_XML",
    "CHECKPOINT_ROOT",
    "RUNS_ROOT",
    "REPORTS_ROOT",
]
