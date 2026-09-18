"""广义力分解与「肌肉力 / 被动力 / bias / 约束力」的区分。

## 单位警告（必须遵守）

``data.qfrc_*`` 是**广义力**：平移自由度的分量是力（N），铰链/转动自由度的分量是力矩
（N·m）。把整条向量的**欧氏范数**解释成「承重」「贡献占比」是**量纲错误**。本模块
因此只报告：

* 各分量的范数（作为**量级与突变检测**用，不做占比解释）；
* 约束力按**约束类型**的逐自由度分解（这是可以做到无歧义的），
  以及分解残差（残差必须接近 0，否则说明分解错，而不是「说明力矩被 equality 吸收」）。

## 四类力的来源

=========================  ==============================================================
项                          物理来源
=========================  ==============================================================
``qfrc_actuator``           肌肉执行器产生的广义力（``actuator_force × moment``）
``qfrc_passive``            关节被动项：弹簧刚度 + 关节阻尼（``jnt_stiffness`` / ``dof_damping``）
``qfrc_bias``               科里奥利 / 离心项 **加上重力**（``mj_rne`` 的非惯性部分）
``qfrc_constraint``         接触力 + 关节/腱限位 + equality 约束（**三者的合计**）
``qfrc_applied``            人为施加的广义力（本研究恒为 0，用于自证无外力）
``xfrc_applied``            人为施加的笛卡尔外力（本研究恒为 0）
=========================  ==============================================================

``qfrc_constraint`` **不能**整体归因于 equality。本模块用 ``data.efc_type`` 把它按
``mjCNSTR_*`` 类型拆开，从而给出「接触 / 限位 / equality」各自的实际贡献。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

#: 广义力向量的范数（仅作量级参考，**不可**解释为承重占比）
FORCE_TERM_KEYS = (
    "qfrc_actuator",
    "qfrc_passive",
    "qfrc_bias",
    "qfrc_constraint",
    "qfrc_applied",
    "xfrc_applied",
    "qfrc_inverse",
    "qacc",
)

#: 约束类型编号 → 名称（``mujoco.mjtConstraint``）
_CONSTRAINT_NAMES = {
    0: "equality",
    1: "friction_dof",
    2: "friction_tendon",
    3: "limit_joint",
    4: "limit_tendon",
    5: "contact_frictionless",
    6: "contact_pyramidal",
    7: "contact_elliptic",
}

#: 归并到三大类
_CONSTRAINT_GROUPS = {
    "equality": ("equality",),
    "limit": ("limit_joint", "limit_tendon"),
    "contact": ("contact_frictionless", "contact_pyramidal", "contact_elliptic"),
    "friction": ("friction_dof", "friction_tendon"),
}


def generalized_force_terms(model, data, keys=FORCE_TERM_KEYS) -> Dict[str, float]:
    """各广义力分量的范数与极值（**不做占比解释**）。"""
    out: Dict[str, float] = {}
    for k in keys:
        arr = getattr(data, k, None)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        out[f"{k}_norm"] = float(np.linalg.norm(a))
        out[f"{k}_absmax"] = float(np.max(np.abs(a))) if a.size else 0.0
    out["n_translational_dofs"] = int(np.sum(np.asarray(model.jnt_type) == 0))
    out["unit_note"] = (
        "qfrc_* 为广义力，混合 N 与 N·m；范数只用于量级/突变检测，"
        "不能解释为承重或贡献占比"
    )
    return out


def decompose_constraint_forces(
    model,
    data,
    *,
    verify: bool = True,
) -> Dict[str, Any]:
    """把 ``qfrc_constraint`` 按约束类型拆解为接触 / 限位 / equality / 摩擦。

    做法：对第 i 个约束，其广义力贡献为 ``±J_i^T · efc_force[i]``，其中 ``J_i`` 是
    ``efc_J`` 的第 i 行（稀疏存储：``efc_J_rowadr`` / ``efc_J_rownnz`` / ``efc_J_colind``）。
    符号约定由**数值验证**确定：把全部约束求和后与 ``data.qfrc_constraint`` 比对，
    取使残差最小的符号。

    Returns:
        ``per_group``（组名 → 范数）、``per_type``（类型名 → 范数）、``residual`` 等。
    """
    nv = int(model.nv)
    per_type: Dict[str, np.ndarray] = {name: np.zeros(nv) for name in _CONSTRAINT_NAMES.values()}
    nefc = int(data.nefc)
    efc_type = np.asarray(data.efc_type, dtype=int)
    efc_force = np.asarray(data.efc_force, dtype=float)
    efc_J = np.asarray(data.efc_J, dtype=float)
    rowadr = np.asarray(data.efc_J_rowadr, dtype=int)
    rownnz = np.asarray(data.efc_J_rownnz, dtype=int)
    colind = np.asarray(data.efc_J_colind, dtype=int)

    for i in range(nefc):
        t = int(efc_type[i])
        name = _CONSTRAINT_NAMES.get(t, f"unknown_{t}")
        f = float(efc_force[i])
        a = int(rowadr[i])
        n = int(rownnz[i])
        cols = colind[a : a + n]
        vals = efc_J[a : a + n]
        np.add.at(per_type[name], cols, f * vals)

    total = np.zeros(nv)
    for v in per_type.values():
        total += v

    target = np.asarray(data.qfrc_constraint, dtype=float)
    res_plus = float(np.max(np.abs(total - target)))
    res_minus = float(np.max(np.abs(-total - target)))
    sign = 1.0 if res_plus <= res_minus else -1.0
    if sign < 0:
        for k in per_type:
            per_type[k] = -per_type[k]
        total = -total
    residual = float(np.max(np.abs(total - target)))

    per_group: Dict[str, float] = {}
    for group, members in _CONSTRAINT_GROUPS.items():
        acc = np.zeros(nv)
        for m in members:
            acc += per_type.get(m, np.zeros(nv))
        per_group[group] = float(np.linalg.norm(acc))

    return {
        "nefc": nefc,
        "sign_convention": "+J^T·f" if sign > 0 else "-J^T·f",
        "per_type_norm": {k: float(np.linalg.norm(v)) for k, v in per_type.items()},
        "per_type_absmax": {k: float(np.max(np.abs(v))) if v.size else 0.0 for k, v in per_type.items()},
        "per_group_norm": per_group,
        "target_qfrc_constraint_norm": float(np.linalg.norm(target)),
        "decomposition_residual_absmax": residual,
        "decomposition_ok": bool(residual <= 1e-6 * max(1.0, float(np.max(np.abs(target))))),
        "n_active_by_type": _count_by_type(efc_type),
        "note": (
            "qfrc_constraint 是接触/限位/equality/摩擦的合计；本表按 mjCNSTR_* 类型拆开，"
            "残差接近 0 才说明拆解正确（而不是把整体归因于某一类）"
        ),
    }


def _count_by_type(efc_type: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in np.unique(efc_type):
        out[_CONSTRAINT_NAMES.get(int(t), f"unknown_{int(t)}")] = int(np.sum(efc_type == t))
    return out


def muscle_active_passive_split(model, data, group_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """把肌肉执行器力拆成主动分量与被动分量。

    定义（与 MuJoCo muscle 模型一致）：

    * ``F(act)``：当前状态下的 ``actuator_force``；
    * ``F_passive = F(act=0)``：同一 ``qpos``/``qvel`` 下把激活置零后的力，
      即肌纤维的被动弹性（并联弹性 + 阻尼）分量；
    * ``F_active = F(act) - F(act=0)``：主动分量。

    实现上给 ``MjData`` 做一份副本并把 ``act`` 置零后 ``mj_forward``，
    因此不改变调用方持有的 ``data``。

    Args:
        group_indices: 只统计这些执行器下标；None 表示全部。

    Returns:
        各分量的均值/绝对均值/最大绝对值。
    """
    import mujoco

    act_now = np.asarray(data.act, dtype=float).copy()
    force_now = np.asarray(data.actuator_force, dtype=float).copy()

    probe = mujoco.MjData(model)
    probe.qpos[:] = np.asarray(data.qpos)
    probe.qvel[:] = np.asarray(data.qvel)
    probe.ctrl[:] = np.asarray(data.ctrl)
    probe.act[:] = 0.0
    mujoco.mj_forward(model, probe)
    force_passive = np.asarray(probe.actuator_force, dtype=float).copy()

    idx = slice(None) if group_indices is None else np.asarray(group_indices, dtype=int)
    fa = force_now[idx] - force_passive[idx]
    fp = force_passive[idx]
    ft = force_now[idx]

    def stats(a: np.ndarray) -> Dict[str, float]:
        return {
            "mean_abs": float(np.mean(np.abs(a))) if a.size else 0.0,
            "abs_max": float(np.max(np.abs(a))) if a.size else 0.0,
            "mean_signed": float(np.mean(a)) if a.size else 0.0,
        }

    return {
        "n_actuators": int(np.size(ft)),
        "total_force": stats(ft),
        "active_force": stats(fa),
        "passive_force": stats(fp),
        "active_matches_definition_absmax": float(np.max(np.abs((fa + fp) - ft))) if ft.size else 0.0,
        "act_mean": float(np.mean(act_now[idx])) if np.size(act_now[idx]) else 0.0,
        "definition": "F_active = F(act) - F(act=0)；F_passive = F(act=0)",
    }


__all__ = [
    "generalized_force_terms",
    "decompose_constraint_forces",
    "muscle_active_passive_split",
    "FORCE_TERM_KEYS",
]
