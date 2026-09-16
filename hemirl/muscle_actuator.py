"""肌肉最大等长力 (F0) 参数化：基准快照 + 组倍率，禁止累乘。

## F0 在哪里

MS-Human-700 的每块肌肉是 MuJoCo 的 ``general`` 执行器，``class="muscle"`` 使其
``dyntype/gaintype/biastype`` 均为 MUSCLE，并显式给出::

    <general name="X" class="muscle" tendon="X_tendon"
             lengthrange="lo hi" gainprm="0.75 1.05 F0" biasprm="0.75 1.05 F0" />

``gainprm`` 实际长度 10，前 9 个有效槽位语义为
``range[0], range[1], force(F0), scale, lmin, lmax, vmax, fpmax, fvmax``
（``gainprm[3:9]`` 与 MuJoCo 默认值 200 / 0.5 / 1.6 / 1.5 / 1.3 / 1.2 逐一吻合，
可作为交叉证据）。因此 **F0 位于 gainprm[2]，被动力通道位于 biasprm[2]**。

本模块不依赖文档假设，而是提供 :func:`identify_f0_slots` 做**扰动-响应实证**：
在固定姿态/速度/激活下逐个缩放参数槽位，观察 ``data.actuator_force`` 的响应比。

## 两种缩放模式

* ``active_only``：只缩放 ``gainprm[2]``。肌肉**主动产力能力下降**，被动力（并联弹性）不变。
* ``active_and_passive``：同时缩放 ``gainprm[2]`` 与 ``biasprm[2]``。主动与被动力共同缩放。

两者语义不同，实验时必须记录实际使用的模式。

## 不做什么

* 不改 ``lengthrange``、``scale``、``lmin/lmax``、``timeconst`` 等长度/时间参数。
* 不缩放关节刚度、阻尼或策略动作幅值——那些不是最大等长力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hemirl import muscle_groups
from hemirl.muscle_groups import MuscleMap

#: F0 在 `actuator_gainprm` / `actuator_biasprm` 中的列索引（实证见 identify_f0_slots）
F0_GAIN_SLOT = 2
F0_BIAS_SLOT = 2

#: 前 9 个有效槽位的语义（用于日志与审计）
PRM_SLOT_NAMES: Tuple[str, ...] = (
    "range[0]",
    "range[1]",
    "force(F0)",
    "scale",
    "lmin",
    "lmax",
    "vmax",
    "fpmax",
    "fvmax",
)

MODE_ACTIVE_ONLY = "active_only"
MODE_ACTIVE_AND_PASSIVE = "active_and_passive"
MODES = (MODE_ACTIVE_ONLY, MODE_ACTIVE_AND_PASSIVE)


# ------------------------------------------------------------------ 实证识别


def _actuator_force(model, qpos: np.ndarray, act_value: float) -> np.ndarray:
    """在给定 qpos / 激活下返回执行器力（纯 MuJoCo 前向，ground truth）。"""
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.act[:] = act_value
    mujoco.mj_forward(model, data)
    return np.asarray(data.actuator_force).copy()


def identify_f0_slots(model, factor: float = 1.1, base_qpos: Optional[np.ndarray] = None) -> Dict:
    """扰动-响应实证：确定主动力/被动力通道中 F0 所在的参数列。

    Args:
        model: MjModel（本函数会就地修改并在结束时恢复）。
        factor: 扰动倍率，默认 1.1（+10%）。
        base_qpos: 基准姿态；None 时用 ``model.key_qpos[0]``。

    Returns:
        含 ``active_gain_slot`` / ``passive_bias_slot`` 与逐槽位响应比的字典。
    """
    qpos = np.asarray(model.key_qpos[0] if base_qpos is None else base_qpos).copy()
    gain0 = np.asarray(model.actuator_gainprm).copy()
    bias0 = np.asarray(model.actuator_biasprm).copy()

    def restore() -> None:
        model.actuator_gainprm[:] = gain0
        model.actuator_biasprm[:] = bias0

    try:
        # --- 主动通道：act = 1 ---
        f_ref = _actuator_force(model, qpos, 1.0)
        active: Dict[int, Optional[float]] = {}
        for k in range(gain0.shape[1]):
            if not (gain0[:, k] != 0).any():
                continue
            restore()
            model.actuator_gainprm[:, k] *= factor
            f = _actuator_force(model, qpos, 1.0)
            mask = np.abs(f_ref) > 1e-9
            active[k] = float(np.mean(f[mask] / f_ref[mask])) if mask.any() else None
        restore()

        # --- 被动通道：act = 0 ---
        p_ref = _actuator_force(model, qpos, 0.0)
        passive: Dict[int, Optional[float]] = {}
        for k in range(bias0.shape[1]):
            if not (bias0[:, k] != 0).any():
                continue
            restore()
            model.actuator_biasprm[:, k] *= factor
            f = _actuator_force(model, qpos, 0.0)
            mask = np.abs(p_ref) > 1e-9
            passive[k] = float(np.mean(f[mask] / p_ref[mask])) if mask.any() else None
        restore()

        def pick(table: Mapping[int, Optional[float]]) -> Optional[int]:
            cands = [(k, v) for k, v in table.items() if v is not None]
            if not cands:
                return None
            # 与 factor 最接近者即该通道的缩放槽位
            k, _ = min(cands, key=lambda kv: abs(kv[1] - factor))
            return k

        active_slot = pick(active)
        passive_slot = pick(passive)

        return {
            "factor": factor,
            "qpos_source": "key_qpos[0]" if base_qpos is None else "given",
            "prm_slot_names": PRM_SLOT_NAMES,
            "active_response_gainprm": {PRM_SLOT_NAMES[k]: v for k, v in active.items()},
            "passive_response_biasprm": {PRM_SLOT_NAMES[k]: v for k, v in passive.items()},
            "active_gain_slot": active_slot,
            "passive_bias_slot": passive_slot,
            "active_slot_name": PRM_SLOT_NAMES[active_slot] if active_slot is not None else None,
            "passive_slot_name": PRM_SLOT_NAMES[passive_slot] if passive_slot is not None else None,
            "n_actuators_with_nonzero_active_force": int((np.abs(f_ref) > 1e-9).sum()),
            "n_actuators_with_nonzero_passive_force": int((np.abs(p_ref) > 1e-9).sum()),
        }
    finally:
        restore()


def assert_f0_slots(model) -> None:
    """确认 F0 槽位与模块常量一致，不一致直接抛错。"""
    rep = identify_f0_slots(model)
    if rep["active_gain_slot"] != F0_GAIN_SLOT:
        raise AssertionError(
            f"主动力 F0 槽位实测为 {rep['active_gain_slot']}，与常量 {F0_GAIN_SLOT} 不符: {rep}"
        )
    if rep["passive_bias_slot"] != F0_BIAS_SLOT:
        raise AssertionError(
            f"被动力 F0 槽位实测为 {rep['passive_bias_slot']}，与常量 {F0_BIAS_SLOT} 不符: {rep}"
        )


# ------------------------------------------------------------------ 配置


@dataclass(frozen=True)
class StrengthSpec:
    """患侧肌力参数的完整描述（不可变，便于落盘与复现）。

    Attributes:
        paretic_side: 患侧，``'L'`` 或 ``'R'``。
        upper_scale: 患侧上肢肌力倍率。F0 = upper_scale * F0_baseline。
        lower_scale: 患侧下肢肌力倍率。
        torso_scale: 躯干肌力倍率；默认 1.0（保持基准）。
        group_scales: 额外/覆盖倍率。键可以是组键（``'L/upper'`` / ``'R/lower'`` /
            ``'M/torso'`` / ``'torso'``）或单个肌肉名（``'bflh_r'``）。命名肌肉优先级最高。
        include_shoulder_girdle: 是否把「权威归 torso、但只跨越上肢关节」的肌肉
            （斜方肌、肩胛提肌等，共 44 块）计入上肢组。默认 False，即上肢组 =
            官方 ``Muscle_Arm_*.xml`` 的 61 块肌肉/侧。
        mode: ``active_only`` 或 ``active_and_passive``。
        note: 备注（写入 provenance）。
    """

    paretic_side: str = "R"
    upper_scale: float = 1.0
    lower_scale: float = 1.0
    torso_scale: float = 1.0
    group_scales: Mapping[str, float] = field(default_factory=dict)
    include_shoulder_girdle: bool = False
    mode: str = MODE_ACTIVE_ONLY
    note: str = ""

    def __post_init__(self) -> None:
        if self.paretic_side not in ("L", "R"):
            raise ValueError(f"paretic_side 必须是 'L' 或 'R'，收到 {self.paretic_side!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES}，收到 {self.mode!r}")
        for label, v in (
            ("upper_scale", self.upper_scale),
            ("lower_scale", self.lower_scale),
            ("torso_scale", self.torso_scale),
        ):
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{label} 必须为非负有限数，收到 {v}")
        for k, v in self.group_scales.items():
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"group_scales[{k!r}] 必须为非负有限数，收到 {v}")

    @property
    def healthy_side(self) -> str:
        return "L" if self.paretic_side == "R" else "R"

    def to_dict(self) -> Dict[str, object]:
        d = {
            "paretic_side": self.paretic_side,
            "upper_scale": self.upper_scale,
            "lower_scale": self.lower_scale,
            "torso_scale": self.torso_scale,
            "group_scales": dict(self.group_scales),
            "include_shoulder_girdle": self.include_shoulder_girdle,
            "mode": self.mode,
            "note": self.note,
        }
        return d


@dataclass
class AppliedStrength:
    """一次 apply 之后的实际结果（落盘用）。"""

    spec: Dict[str, object]
    multiplier: np.ndarray  # (nu,) 每块肌肉的 F0 倍率
    n_scaled: int
    n_unchanged: int
    group_counts: Dict[str, int]
    f0_min_multiplier: float
    f0_max_multiplier: float
    applied_slots: Dict[str, int]

    def summary(self) -> Dict[str, object]:
        return {
            "spec": self.spec,
            "n_scaled": self.n_scaled,
            "n_unchanged": self.n_unchanged,
            "group_counts": self.group_counts,
            "f0_multiplier_min": self.f0_min_multiplier,
            "f0_multiplier_max": self.f0_max_multiplier,
            "applied_slots": self.applied_slots,
        }


# ------------------------------------------------------------------ 缩放器


class StrengthScaler:
    """按组设置肌肉最大等长力倍率，始终从不可变基准计算。

    典型用法::

        scaler = StrengthScaler(model, mapping)          # 构造时冻结基准
        scaler.apply(StrengthSpec(paretic_side='R', upper_scale=0.5))
        scaler.apply(StrengthSpec(paretic_side='R', upper_scale=0.75))  # = 0.75 * 基准
        scaler.reset()                                    # 完全恢复基准

    绝不做 ``*=`` 累乘：每次 ``apply`` 都写成 ``baseline * alpha``。
    """

    def __init__(
        self,
        model,
        mapping: MuscleMap,
        mode: str = MODE_ACTIVE_ONLY,
        verify_slots: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES}，收到 {mode!r}")
        self.model = model
        self.mapping = mapping
        self.mode = mode
        # 不可变基准：构造时拷贝，之后只读
        self._gain0 = np.asarray(model.actuator_gainprm).copy()
        self._bias0 = np.asarray(model.actuator_biasprm).copy()
        if verify_slots:
            assert_f0_slots(model)
        self._names: List[str] = list(self._actuator_names())

    # ------------------------------------------------------------ 内部

    def _actuator_names(self) -> Sequence[str]:
        import mujoco

        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
            for i in range(self.model.nu)
        ]

    def baseline_gain(self) -> np.ndarray:
        """基准 ``actuator_gainprm``（只读副本）。"""
        return self._gain0.copy()

    def baseline_bias(self) -> np.ndarray:
        return self._bias0.copy()

    def baseline_f0(self) -> np.ndarray:
        """每块肌肉的基准 F0。"""
        return self._gain0[:, F0_GAIN_SLOT].copy()

    def current_multiplier(self) -> np.ndarray:
        cur = np.asarray(self.model.actuator_gainprm)[:, F0_GAIN_SLOT]
        base = self._gain0[:, F0_GAIN_SLOT]
        with np.errstate(divide="ignore", invalid="ignore"):
            mult = np.where(base != 0, cur / base, 1.0)
        return mult

    def group_indices(self, include_shoulder_girdle: bool = False) -> Dict[str, List[int]]:
        """返回各组的执行器索引（组键见 :func:`group_keys`）。"""
        m = self.mapping
        groups: Dict[str, List[int]] = {}
        for side in ("L", "R"):
            for limb in ("upper", "lower", "torso"):
                groups[f"{side}/{limb}"] = m.indices_for(side, limb)
        groups["M/torso"] = m.indices_for("M", "torso")
        if include_shoulder_girdle:
            sg = self.shoulder_girdle_indices()
            groups["R/upper"] = sorted(set(groups["R/upper"]) | set(sg["R"]))
            groups["L/upper"] = sorted(set(groups["L/upper"]) | set(sg["L"]))
        return groups

    def shoulder_girdle_indices(self) -> Dict[str, List[int]]:
        """权威归 torso、但只跨越上肢关节的肌肉（斜方肌、肩胛提肌等）。"""
        out: Dict[str, List[int]] = {"L": [], "R": [], "M": []}
        for e in self.mapping.entries:
            if e.limb_group == "torso" and e.crossed_limbs == ["upper"]:
                out.setdefault(e.side, []).append(e.index)
        return out

    # ------------------------------------------------------------ 应用

    def multipliers_for(self, spec: StrengthSpec) -> np.ndarray:
        """计算每块肌肉的 F0 倍率（纯函数，不触碰模型）。"""
        nu = self.model.nu
        mult = np.ones(nu, dtype=np.float64)

        groups = self.group_indices(include_shoulder_girdle=spec.include_shoulder_girdle)

        # 1) 基础：患侧肢体的上下肢倍率
        up_key = f"{spec.paretic_side}/upper"
        lo_key = f"{spec.paretic_side}/lower"
        for i in groups.get(up_key, []):
            mult[i] *= spec.upper_scale
        for i in groups.get(lo_key, []):
            mult[i] *= spec.lower_scale

        # 2) 躯干倍率
        if spec.torso_scale != 1.0:
            for side in ("L", "R", "M"):
                for i in groups.get(f"{side}/torso", []):
                    mult[i] *= spec.torso_scale

        # 3) 显式组键覆盖（整组替换为该倍率）
        for key, alpha in spec.group_scales.items():
            if key in groups:
                for i in groups[key]:
                    mult[i] = alpha

        # 4) 显式肌肉名覆盖（最高优先级）
        name_to_index = {n: i for i, n in enumerate(self._names)}
        for key, alpha in spec.group_scales.items():
            if key in name_to_index:
                mult[name_to_index[key]] = alpha

        unknown = [
            k
            for k in spec.group_scales
            if k not in groups and k not in name_to_index
        ]
        if unknown:
            raise KeyError(
                f"group_scales 中出现未知组键/肌肉名: {unknown}；"
                f"可用组键: {sorted(groups)}（另可用具体肌肉名）"
            )
        return mult

    def apply(self, spec: StrengthSpec) -> AppliedStrength:
        """把倍率写入模型（从基准计算，绝不累乘）。"""
        mult = self.multipliers_for(spec)
        gain = self._gain0.copy()
        bias = self._bias0.copy()

        gain[:, F0_GAIN_SLOT] = self._gain0[:, F0_GAIN_SLOT] * mult
        if spec.mode == MODE_ACTIVE_AND_PASSIVE:
            bias[:, F0_BIAS_SLOT] = self._bias0[:, F0_BIAS_SLOT] * mult

        self.model.actuator_gainprm[:] = gain
        self.model.actuator_biasprm[:] = bias

        counts: Dict[str, int] = {}
        for i, a in enumerate(mult):
            if a != 1.0:
                entry = self.mapping.entries[i]
                counts[entry.group_key] = counts.get(entry.group_key, 0) + 1

        return AppliedStrength(
            spec=spec.to_dict(),
            multiplier=mult,
            n_scaled=int((mult != 1.0).sum()),
            n_unchanged=int((mult == 1.0).sum()),
            group_counts=counts,
            f0_min_multiplier=float(mult.min()),
            f0_max_multiplier=float(mult.max()),
            applied_slots={
                "gainprm": F0_GAIN_SLOT,
                "biasprm": F0_BIAS_SLOT if spec.mode == MODE_ACTIVE_AND_PASSIVE else -1,
            },
        )

    def reset(self) -> None:
        """恢复全部基准参数。"""
        self.model.actuator_gainprm[:] = self._gain0.copy()
        self.model.actuator_biasprm[:] = self._bias0.copy()

    # ------------------------------------------------------------ 校验

    def assert_baseline(self, atol: float = 0.0) -> None:
        """确认当前模型参数等于基准。"""
        g = np.asarray(self.model.actuator_gainprm)
        b = np.asarray(self.model.actuator_biasprm)
        if not np.allclose(g, self._gain0, atol=atol, rtol=0):
            raise AssertionError(
                f"gainprm 与基准不一致，最大偏差 {np.max(np.abs(g - self._gain0))}"
            )
        if not np.allclose(b, self._bias0, atol=atol, rtol=0):
            raise AssertionError(
                f"biasprm 与基准不一致，最大偏差 {np.max(np.abs(b - self._bias0))}"
            )

    def assert_group_unchanged(self, indices: Sequence[int], atol: float = 0.0) -> None:
        """确认指定执行器参数等于基准。"""
        g = np.asarray(self.model.actuator_gainprm)[list(indices)]
        g0 = self._gain0[list(indices)]
        if not np.allclose(g, g0, atol=atol, rtol=0):
            bad = np.where(np.any(g != g0, axis=1))[0]
            raise AssertionError(f"执行器 {list(np.asarray(indices)[bad])[:10]} 参数被意外修改")


def describe_slots() -> Dict[str, object]:
    """返回 F0 槽位常量及其语义（写报告用）。"""
    return {
        "F0_GAIN_SLOT": F0_GAIN_SLOT,
        "F0_BIAS_SLOT": F0_BIAS_SLOT,
        "prm_slot_names": list(PRM_SLOT_NAMES),
        "modes": list(MODES),
    }


__all__ = [
    "F0_GAIN_SLOT",
    "F0_BIAS_SLOT",
    "PRM_SLOT_NAMES",
    "MODE_ACTIVE_ONLY",
    "MODE_ACTIVE_AND_PASSIVE",
    "StrengthSpec",
    "StrengthScaler",
    "AppliedStrength",
    "identify_f0_slots",
    "assert_f0_slots",
    "describe_slots",
    "muscle_groups",
]
