"""包装器：直接复用官方 `MuscleNormWrapper`，避免行为漂移。

官方定义位于 ``external/msgym/DynSyn-SAC/SB3-Scripts/wrapper/muscle_norm_wrapper.py``。
本模块把它按路径导入并做**行为等价性自检**（对 [-1,1] 网格逐点比对），
确保我们没有无意间改变 checkpoint 的动作接口。

动作语义::

    a_env = 1 / (1 + exp(-5 * (a_policy - 0.5)))
    a_policy ∈ [-1, 1]  ->  a_env ∈ (0, 1) = 肌肉 excitation
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from hemirl import paths

_CACHE: dict = {}


def official_muscle_norm_wrapper():
    """按路径导入上游 `MuscleNormWrapper` 类。"""
    if "cls" in _CACHE:
        return _CACHE["cls"]
    scripts = str(paths.MSGYM_ROOT / "DynSyn-SAC" / "SB3-Scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from wrapper.muscle_norm_wrapper import MuscleNormWrapper  # type: ignore

    _CACHE["cls"] = MuscleNormWrapper
    return MuscleNormWrapper


def reference_sigmoid(action: np.ndarray) -> np.ndarray:
    """上游映射的参考实现（仅用于自检）。"""
    return 1.0 / (1.0 + np.exp(-5.0 * (np.asarray(action) - 0.5)))


def verify_wrapper_equivalence(n: int = 21, atol: float = 0.0) -> dict:
    """逐点比对上游实现与参考实现，返回差异统计。"""
    cls = official_muscle_norm_wrapper()
    grid = np.linspace(-1.0, 1.0, n)
    diffs = []
    for v in grid:
        got = cls.action(object.__new__(cls), np.array([v]))  # 不经过 __init__ 直接调用 action
        ref = reference_sigmoid(np.array([v]))
        diffs.append(float(np.max(np.abs(got - ref))))
    return {
        "n_points": n,
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "equivalent": bool(max(diffs) <= atol) if diffs else False,
        "sample": {f"{v:.2f}": float(reference_sigmoid(np.array([v]))[0]) for v in [-1.0, -0.5, 0.5, 1.0]},
    }


__all__ = ["official_muscle_norm_wrapper", "reference_sigmoid", "verify_wrapper_equivalence"]
