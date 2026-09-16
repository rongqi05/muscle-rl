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

    # 多个姿态下重复，确认不是单点巧合（每次测量前都 reset 取基准）
    ratios_multi = []
    rng = np.random.default_rng(0)
    for _ in range(3):
        q = q_key + rng.normal(0, 0.15, size=model.nq)
        q[3:7] = q_key[3:7]  # 保持根节点四元数合法
        scaler.reset()
        fa = force_at(q, 1.0)
        scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.25, lower_scale=1.0))
        fb = force_at(q, 1.0)
        idx = np.array(sorted(mapping.indices_for("R", "upper")))
        m = np.abs(fa[idx]) > 1e-3
        if m.any():
            ratios_multi.append(float(np.mean(fb[idx][m] / fa[idx][m])))
    ck.check(
        "V5c 三个随机姿态下患侧上肢力比仍 ≈ 0.25",
        len(ratios_multi) == 3 and all(abs(r - 0.25) < 0.03 for r in ratios_multi),
        {"ratios": [round(r, 4) for r in ratios_multi]},
    )

    # ---------------------------------------------------------- V6
    # 等长力-长度曲线：改变 qpos 幅度，检查缩放前后比值是否恒定
    scaler.reset()
    curve_base, curve_scaled = [], []
    amps = [0.0, 0.2, 0.4, 0.6]
    for amp in amps:
        q = q_key.copy()
        q[7:] = q_key[7:] * (1.0 + amp)
        scaler.reset()
        fb = force_at(q, 1.0)
        scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, lower_scale=1.0))
        fs = force_at(q, 1.0)
        idx = np.array(sorted(mapping.indices_for("R", "upper")))
        m = np.abs(fb[idx]) > 1e-3
        curve_base.append(float(np.abs(fb[idx][m]).mean()) if m.any() else 0.0)
        curve_scaled.append(float(np.mean(fs[idx][m] / fb[idx][m])) if m.any() else np.nan)
    ck.check(
        "V6 不同长度下缩放比恒定（只改幅值，不改 FL 形状）",
        all((not np.isnan(r)) and abs(r - 0.5) < 0.03 for r in curve_scaled),
        {"amplitudes": amps, "baseline_mean_abs_force": [round(v, 3) for v in curve_base],
         "ratios": [round(float(r), 4) for r in curve_scaled]},
    )

    # ---------------------------------------------------------- V7
    # 被动通道：act = 0
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
