"""肌力参数化的验证脚本（对应任务书第七节）。

覆盖：
 V1 倍率 1.0 恢复基准参数
 V2 连续设置 0.5 再 0.75 == 基准的 0.75（无累乘）
 V3 左右侧 / 上下肢映射正确
 V4 指定组以外的肌肉参数不变
 V5 相同姿态/速度/激活下，主动产力参数变化产生预期效果
 V6 主动力缩放不改变等长力-长度曲线的形状（只改幅值）
 V7 两种模式（active_only / active_and_passive）对被动力的影响可区分

用法::

    PYTHONPATH=. python scripts/verify_strength.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import muscle_actuator as ma  # noqa: E402
from hemirl import muscle_groups, paths  # noqa: E402


class Checker:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def check(self, name: str, ok: bool, detail: object = None) -> None:
        self.results.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  | {detail}" if detail is not None else ""))

    def summary(self) -> dict:
        n_ok = sum(1 for r in self.results if r["ok"])
        return {"n_checks": len(self.results), "n_pass": n_ok, "all_pass": n_ok == len(self.results),
                "checks": self.results}


# ------------------------------------------------------------------ 姿态与测量辅助


def _state_at(model, qpos: np.ndarray, act: float):
    """在给定 ``qpos``（``qvel=0``）与恒定激活下做一次前向，返回 (力, 长度)。

    用 ``mj_forward`` 而不是 ``mj_step``，因此这是**等长**测量：长度由姿态完全决定，
    速度为 0，主动力唯一地来自激活与 F0。
    """
    import mujoco

    d = mujoco.MjData(model)
    d.qpos[:] = np.asarray(qpos, dtype=float)
    d.qvel[:] = 0.0
    d.act[:] = float(act)
    mujoco.mj_forward(model, d)
    return np.asarray(d.actuator_force).copy(), np.asarray(d.actuator_length).copy()


def _arm_postures(model, q_key: np.ndarray) -> dict:
    """按**关节名**构造患侧（R）肩 / 肘姿态。

    **不使用任何索引假设**：本模型没有 freejoint（``nq == nv == 85``，骨盆由 3 个 slide
    + 3 个 hinge 组成），所以不存在「根节点四元数」需要保护；按名字改自由度才是稳定的做法。

    偏移量均落在 ``jnt_range`` 内（关键姿态下这些角度为 0）：
    ``shoulder_elv_r ∈ [0, 3.1]``、``elbow_flexion_r ∈ [0, 2.2]``、
    ``elv_angle_r ∈ [-1.6, 2.2]``、``shoulder_rot_r ∈ [-0.8, 1.6]``。
    """
    out = {"key": np.asarray(q_key, dtype=float).copy()}

    def with_offsets(offsets):
        q = np.asarray(q_key, dtype=float).copy()
        for jname, delta in offsets.items():
            adr = int(model.joint(jname).qposadr[0])
            q[adr] = float(q_key[adr]) + delta
        return q

    out["shoulder_elv"] = with_offsets({"shoulder_elv_r": 0.60})
    out["elbow_flex"] = with_offsets({"elbow_flexion_r": 1.00})
    out["combined"] = with_offsets(
        {
            "shoulder_elv_r": 0.50,
            "elv_angle_r": 0.40,
            "shoulder_rot_r": 0.30,
            "elbow_flexion_r": 1.00,
        }
    )
    return out


def _muscles_crossing(mapping, joints, side: str, limb: str):
    """返回属于 ``side/limb`` 组且跨越 ``joints`` 中任一关节的肌肉名。"""
    idx = set(mapping.indices_for(side, limb))
    out = []
    for e in mapping.entries:
        if e.index not in idx:
            continue
        if any(j in (e.crossed_joints or []) for j in joints):
            out.append(e.name)
    return out


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=str(paths.MODEL_XML))
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "verify_strength.json"))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    mapping = muscle_groups.build_map(model)
    ck = Checker()

    # ---------------------------------------------------------- 槽位实证
    slot_rep = ma.identify_f0_slots(model)
    ck.check(
        "V0 F0 槽位实测 == gainprm[2]/biasprm[2]",
        slot_rep["active_gain_slot"] == ma.F0_GAIN_SLOT and slot_rep["passive_bias_slot"] == ma.F0_BIAS_SLOT,
        {
            "active": slot_rep["active_slot_name"],
            "passive": slot_rep["passive_slot_name"],
            "active_ratios": {k: round(v, 4) if v else v for k, v in slot_rep["active_response_gainprm"].items()},
            "passive_ratios": {k: round(v, 4) if v else v for k, v in slot_rep["passive_response_biasprm"].items()},
            "n_active_force_nonzero": slot_rep["n_actuators_with_nonzero_active_force"],
            "n_passive_force_nonzero": slot_rep["n_actuators_with_nonzero_passive_force"],
        },
    )

    scaler = ma.StrengthScaler(model, mapping, verify_slots=False)
    nu = model.nu
    gauge = scaler.baseline_gain()
    biaseg = scaler.baseline_bias()

    def force_at(qpos: np.ndarray, act: float) -> np.ndarray:
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.act[:] = act
        mujoco.mj_forward(model, data)
        return np.asarray(data.actuator_force).copy()

    q_key = np.asarray(model.key_qpos[0]).copy()

    # ---------------------------------------------------------- V1
    spec_half = ma.StrengthSpec(paretic_side="R", upper_scale=0.5)
    scaler.apply(spec_half)
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=1.0))
    g = np.asarray(model.actuator_gainprm)
    b = np.asarray(model.actuator_biasprm)
    ck.check(
        "V1 倍率 1.0 精确恢复基准参数",
        np.array_equal(g, gauge) and np.array_equal(b, biaseg),
        {"max_abs_diff_gain": float(np.max(np.abs(g - gauge))), "max_abs_diff_bias": float(np.max(np.abs(b - biaseg)))},
    )

    # ---------------------------------------------------------- V2
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, lower_scale=0.5))
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.75, lower_scale=0.75))
    cur = np.asarray(model.actuator_gainprm)[:, ma.F0_GAIN_SLOT]
    base = gauge[:, ma.F0_GAIN_SLOT]
    with np.errstate(divide="ignore", invalid="ignore"):
        mult = np.where(base != 0, cur / base, 1.0)
    scaled_idx = np.where(mult != 1.0)[0]
    ck.check(
        "V2 连续 0.5 → 0.75 结果 == 基准 × 0.75（无累乘）",
        bool(np.allclose(mult[scaled_idx], 0.75, rtol=1e-12, atol=1e-12)) and scaled_idx.size > 0,
        {"n_scaled": int(scaled_idx.size), "unique_multipliers": sorted(set(np.round(mult, 12).tolist())),
         "max_dev_from_0.75": float(np.max(np.abs(mult[scaled_idx] - 0.75))) if scaled_idx.size else None},
    )

    # ---------------------------------------------------------- V3
    scaler.reset()
    spec = ma.StrengthSpec(paretic_side="R", upper_scale=0.4, lower_scale=1.0)
    applied = scaler.apply(spec)
    got = set(np.where(applied.multiplier != 1.0)[0].tolist())
    expect_upper = set(mapping.indices_for("R", "upper"))
    ck.check(
        "V3a 患侧=R 上肢 0.4 → 恰好缩放 R/upper",
        got == expect_upper,
        {"n_got": len(got), "n_expect": len(expect_upper),
         "extra": sorted(got - expect_upper)[:10], "missing": sorted(expect_upper - got)[:10]},
    )
    # 上下肢独立：下肢组倍率 1.0 时不应被改变
    scaler.reset()
    spec2 = ma.StrengthSpec(paretic_side="R", upper_scale=1.0, lower_scale=0.4)
    applied2 = scaler.apply(spec2)
    got2 = set(np.where(applied2.multiplier != 1.0)[0].tolist())
    expect_lower = set(mapping.indices_for("R", "lower"))
    ck.check(
        "V3b 患侧=R 下肢 0.4 → 恰好缩放 R/lower",
        got2 == expect_lower,
        {"n_got": len(got2), "n_expect": len(expect_lower)},
    )
    ck.check(
        "V3c 上下肢集合不相交且覆盖两侧肢体",
        len(expect_upper & expect_lower) == 0
        and len(expect_upper) == len(mapping.indices_for("L", "upper"))
        and len(expect_lower) == len(mapping.indices_for("L", "lower")),
        {"n_R_upper": len(expect_upper), "n_R_lower": len(expect_lower),
         "n_L_upper": len(mapping.indices_for("L", "upper")),
         "n_L_lower": len(mapping.indices_for("L", "lower"))},
    )
    # 侧别对称性：L 患侧时缩放的应是 L 组
    scaler.reset()
    applied_L = scaler.apply(ma.StrengthSpec(paretic_side="L", upper_scale=0.4, lower_scale=0.4))
    got_L = set(np.where(applied_L.multiplier != 1.0)[0].tolist())
    ck.check(
        "V3d 患侧=L → 缩放 L/upper ∪ L/lower，患对侧完全不变",
        got_L == set(mapping.indices_for("L", "upper")) | set(mapping.indices_for("L", "lower")),
        {"n_got": len(got_L), "n_R_scaled": len([i for i in got_L if i in set(mapping.indices_for('R','upper')) | set(mapping.indices_for('R','lower'))])},
    )

    # ---------------------------------------------------------- V4
    scaler.reset()
    applied = scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.25, lower_scale=0.5))
    others = sorted(
        set(range(nu)) - set(mapping.indices_for("R", "upper")) - set(mapping.indices_for("R", "lower"))
    )
    g = np.asarray(model.actuator_gainprm)
    b = np.asarray(model.actuator_biasprm)
    ok_gain = np.array_equal(g[others], gauge[others])
    ok_bias = np.array_equal(b[others], biaseg[others])
    ck.check(
        "V4 指定组以外的肌肉参数逐元素不变（含健侧、躯干）",
        ok_gain and ok_bias,
        {"n_unscaled": len(others), "gain_equal": ok_gain, "bias_equal": ok_bias},
    )

    # ---------------------------------------------------------- V5
    # 注意：测量基准力之前必须先 reset，否则拿到的是上一次 apply 后的状态。
    scaler.reset()
    f0_baseline = force_at(q_key, 1.0)
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, lower_scale=1.0))
    f0_scaled = force_at(q_key, 1.0)
    idx = np.array(sorted(mapping.indices_for("R", "upper")))
    mask = np.abs(f0_baseline[idx]) > 1e-6
    ratio = f0_scaled[idx][mask] / f0_baseline[idx][mask]
    ck.check(
        "V5 同姿态/速度/激活下，患侧上肢主动力按倍率变化（active_only）",
        mask.sum() > 0 and np.allclose(ratio, 0.5, atol=0.02),
        {"n_measured": int(mask.sum()), "ratio_mean": float(ratio.mean()),
         "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max())},
    )
    idx_lo = np.array(sorted(mapping.indices_for("R", "lower")))
    mask_lo = np.abs(f0_baseline[idx_lo]) > 1e-6
    ratio_lo = f0_scaled[idx_lo][mask_lo] / f0_baseline[idx_lo][mask_lo]
    ck.check(
        "V5b 下肢倍率 1.0 时下肢力完全不变",
        bool(np.allclose(ratio_lo, 1.0, rtol=0, atol=0)),
        {"n_measured": int(mask_lo.sum())},
    )

    # 多个姿态下重复，确认不是单点巧合（每次测量前都 reset 取基准）。
    #
    # 注意：本模型**没有 freejoint**（nq == nv == 85；骨盆是 3 个 slide + 3 个 hinge），
    # 因此不存在需要保护的「根节点四元数」。旧版本里 `q[3:7] = key[3:7]` 的做法来自
    # 对 freejoint 布局的假设，在本模型上既无意义又掩盖了真实自由度。
    # 姿态一律按**关节名**构造（见 `_arm_postures`），不使用任何索引假设。
    postures = _arm_postures(model, q_key)
    ratios_multi = []
    for posture_name, q in postures.items():
        scaler.reset()
        fa = force_at(q, 1.0)
        scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.25, lower_scale=1.0))
        fb = force_at(q, 1.0)
        idx = np.array(sorted(mapping.indices_for("R", "upper")))
        m = np.abs(fa[idx]) > 1e-3
        if m.any():
            ratios_multi.append((posture_name, float(np.mean(fb[idx][m] / fa[idx][m]))))
    ck.check(
        "V5c 多个（按关节名构造的）姿态下患侧上肢力比仍 ≈ 0.25",
        len(ratios_multi) == len(postures) and all(abs(r - 0.25) < 0.03 for _, r in ratios_multi),
        {"postures": [p for p, _ in ratios_multi], "ratios": [round(r, 4) for _, r in ratios_multi]},
    )

    # ---------------------------------------------------------- V6
    # 主动力缩放验证。旧版本用「放大初始关节角」改变姿态，四组基准上肢肌肉力完全一致，
    # 说明那些肌肉的长度根本没变（= 没有真正测到 FL 曲线），因此结论是空的。
    # 现在按**关节名**选肩/肘，并分三步：
    #   (a) 先证明目标肌肉的长度**确实改变**；
    #   (b) 再用 F_active(a) = F(a) - F(0) 分离主动分量，验证缩放比恒等于倍率；
    #   (c) 同时给出被动力对照（active_only 不变 / active_and_passive 按倍率变）。
    shoulder_joints = ("elv_angle_r", "shoulder_elv_r", "shoulder_rot_r")
    elbow_joints = ("elbow_flexion_r",)
    shoulder_muscles = _muscles_crossing(mapping, shoulder_joints, side="R", limb="upper")
    elbow_muscles = _muscles_crossing(mapping, elbow_joints, side="R", limb="upper")

    length_evidence: dict = {}
    active_ratios: dict = {}
    passive_evidence: dict = {}
    idx_all = np.array(sorted(mapping.indices_for("R", "upper")))
    name_to_index = {e.name: e.index for e in mapping.entries}

    for posture_name, q in postures.items():
        if posture_name == "key":
            continue

        # (a) 长度确实改变（与关键姿态对照）
        scaler.reset()
        _, length_base = _state_at(model, q, act=0.0)
        _, length_key = _state_at(model, q_key, act=0.0)
        for label, names in (("shoulder", shoulder_muscles), ("elbow", elbow_muscles)):
            sel = [name_to_index[n] for n in names if n in name_to_index]
            sel = [i for i in sel if i in set(idx_all.tolist())]
            if not sel:
                continue
            d_len = np.abs(np.asarray(length_base)[sel] - np.asarray(length_key)[sel])
            length_evidence[f"{posture_name}/{label}"] = {
                "n_muscles": len(sel),
                "max_abs_length_change_m": float(np.max(d_len)) if d_len.size else 0.0,
                "mean_abs_length_change_m": float(np.mean(d_len)) if d_len.size else 0.0,
                "changed": bool(d_len.size and np.max(d_len) > 1e-4),
            }
        ck.check(
            f"V6a[{posture_name}] 目标肌肉长度确实改变（姿态按关节名构造）",
            all(v["changed"] for k, v in length_evidence.items() if k.startswith(posture_name + "/")),
            {k: round(v["max_abs_length_change_m"], 5)
             for k, v in length_evidence.items() if k.startswith(posture_name + "/")},
        )

        # (b) 主动分量 F_active = F(act=1) - F(act=0) 按倍率缩放
        scaler.reset()
        f_act_base, _ = _state_at(model, q, act=1.0)
        scaler.reset()
        f_pas_base, _ = _state_at(model, q, act=0.0)
        scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, lower_scale=1.0, mode=ma.MODE_ACTIVE_ONLY))
        f_act_scaled, _ = _state_at(model, q, act=1.0)
        f_pas_scaled, _ = _state_at(model, q, act=0.0)

        active_base = f_act_base - f_pas_base
        active_scaled = f_act_scaled - f_pas_scaled
        m = np.abs(active_base[idx_all]) > 1e-2  # 主动分量足够大才有意义
        ratio_active = active_scaled[idx_all][m] / active_base[idx_all][m] if m.any() else np.array([])
        active_ratios[posture_name] = {
            "n_measured": int(m.sum()),
            "mean": float(ratio_active.mean()) if ratio_active.size else None,
            "min": float(ratio_active.min()) if ratio_active.size else None,
            "max": float(ratio_active.max()) if ratio_active.size else None,
        }
        ck.check(
            f"V6b[{posture_name}] 同姿态/速度/激活下 F_active 按倍率缩放（0.5）",
            m.sum() > 0 and bool(np.allclose(ratio_active, 0.5, atol=0.02)),
            active_ratios[posture_name],
        )

        # (c) 被动力对照
        pm = np.abs(f_pas_base[idx_all]) > 1e-3
        passive_evidence[posture_name] = {
            "n_nonzero_passive": int(pm.sum()),
            "active_only_unchanged": bool(
                np.array_equal(f_pas_scaled[idx_all][pm], f_pas_base[idx_all][pm])
            ),
        }
        scaler.reset()
        scaler.apply(
            ma.StrengthSpec(
                paretic_side="R", upper_scale=0.5, lower_scale=1.0, mode=ma.MODE_ACTIVE_AND_PASSIVE
            )
        )
        f_pas_both, _ = _state_at(model, q, act=0.0)
        r_both = f_pas_both[idx_all][pm] / f_pas_base[idx_all][pm] if pm.any() else np.array([])
        passive_evidence[posture_name]["active_and_passive_ratio_mean"] = (
            float(r_both.mean()) if r_both.size else None
        )
        passive_evidence[posture_name]["active_and_passive_scales_passive"] = bool(
            r_both.size == 0 or np.allclose(r_both, 0.5, atol=0.02)
        )
    ck.check(
        "V6c 被动力对照：active_only 逐元素不变、active_and_passive 按倍率缩放",
        all(
            v["active_only_unchanged"] and v["active_and_passive_scales_passive"]
            for v in passive_evidence.values()
        ),
        passive_evidence,
    )

    # ---------------------------------------------------------- V7
    # 被动通道（act = 0）：再在关键姿态上确认一次两种模式的区分度
    scaler.reset()
    p_base = force_at(q_key, 0.0)
    scaler.reset()
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, mode=ma.MODE_ACTIVE_ONLY))
    p_active_only = force_at(q_key, 0.0)
    scaler.reset()
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, mode=ma.MODE_ACTIVE_AND_PASSIVE))
    p_both = force_at(q_key, 0.0)

    idx = np.array(sorted(mapping.indices_for("R", "upper")))
    m = np.abs(p_base[idx]) > 1e-6
    ok_active_only = bool(np.allclose(p_active_only[idx][m], p_base[idx][m], rtol=0, atol=0)) if m.any() else None
    r_both = (p_both[idx][m] / p_base[idx][m]) if m.any() else np.array([])
    ck.check(
        "V7a active_only：被动力完全不变",
        ok_active_only is True or m.sum() == 0,
        {"n_nonzero_passive": int(m.sum())},
    )
    ck.check(
        "V7b active_and_passive：被动力按倍率缩放",
        m.sum() == 0 or bool(np.allclose(r_both, 0.5, atol=0.02)),
        {"n_nonzero_passive": int(m.sum()), "ratio_mean": float(r_both.mean()) if r_both.size else None},
    )
    ck.check(
        "V7c 两种模式的被动力可区分",
        ok_active_only is not None and not np.allclose(p_active_only, p_both),
        {"max_abs_diff_passive_force": float(np.max(np.abs(p_active_only - p_both)))},
    )

    # ---------------------------------------------------------- 收尾
    scaler.reset()
    scaler.assert_baseline()
    ck.check("V8 reset 后与基准逐元素一致", True, {"max_abs_diff": 0.0})

    summary = ck.summary()
    summary["slot_identification"] = slot_rep
    summary["muscle_map_counts"] = mapping.counts()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{summary['n_pass']}/{summary['n_checks']} 通过  ->  {out}")
    if not summary["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
