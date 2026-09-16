"""核心单元测试（不依赖 SB3 / 不依赖网络，可直接跑）。

覆盖：
1. 肌群映射：计数与官方源文件一致、左右侧对称、互不重叠。
2. 肌力缩放语义：1.0 还原、无累乘、组外不变、两种模式的被动力差异。
3. F0 槽位：实证结果与常量一致。
4. 终止配置：字段校验、JSON 往返、拒绝「因参考偏差终止」。
5. 策略组展开：138 → 700 的索引语义。
6. numpy 兼容层：幂等。

用法::

    PYTHONPATH=. python scripts/run_tests.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import muscle_actuator as ma  # noqa: E402
from hemirl import muscle_groups, paths  # noqa: E402
from hemirl.numpy_compat import install_numpy2_pickle_compat  # noqa: E402
from hemirl.policy import build_group_expansion, expand_group_action  # noqa: E402
from hemirl.termination import TerminationConfig, evaluate_research_termination  # noqa: E402

RESULTS: list = []


def case(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
                RESULTS.append({"test": name, "ok": True, "error": None})
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                RESULTS.append({"test": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

        wrapper.__name__ = fn.__name__
        return wrapper

    return deco


def _model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(paths.MODEL_XML))


# ---------------------------------------------------------------- 肌群映射


@case("肌群映射: 与官方源文件分组计数一致 (50/50 lower, 61/61 upper, 478 torso)")
def test_group_counts():
    m = muscle_groups.build_map(_model())
    counts = m.counts()
    assert counts["R/lower"] == 50, counts
    assert counts["L/lower"] == 50, counts
    assert counts["R/upper"] == 61, counts
    assert counts["L/upper"] == 61, counts
    torso = counts["R/torso"] + counts["L/torso"] + counts["M/torso"]
    assert torso == 478, counts
    assert len(m) == 700


@case("肌群映射: 左右侧肢体肌群一一对应（同名去后缀）")
def test_left_right_symmetry():
    m = muscle_groups.build_map(_model())
    r_upper = {muscle_groups.strip_side(e.name).lower() for e in m.by_group("R", "upper")}
    l_upper = {muscle_groups.strip_side(e.name).lower() for e in m.by_group("L", "upper")}
    assert r_upper == l_upper, (r_upper ^ l_upper)
    r_lower = {muscle_groups.strip_side(e.name).lower() for e in m.by_group("R", "lower")}
    l_lower = {muscle_groups.strip_side(e.name).lower() for e in m.by_group("L", "lower")}
    assert r_lower == l_lower, (r_lower ^ l_lower)


@case("肌群映射: 组之间互不重叠、并集覆盖全部执行器")
def test_groups_partition():
    m = muscle_groups.build_map(_model())
    idxs = []
    for side in ("L", "R", "M"):
        for limb in ("upper", "lower", "torso"):
            idxs.extend(m.indices_for(side, limb))
    assert len(idxs) == len(set(idxs)), "组之间有重叠"
    assert len(idxs) == 700, len(idxs)


@case("肌群映射: 权威分组均为非 unknown，且无侧别不明的肢体肌肉")
def test_group_consistency():
    m = muscle_groups.build_map(_model())
    problems = muscle_groups.validate_group_consistency(m)
    assert not problems, problems[:5]
    # 「未跨越可动关节」是该简化模型脊柱刚性连接的正常结果，单独报告而非报错
    no_joint = muscle_groups.report_no_movable_joint(m)
    assert len(no_joint) == 192, len(no_joint)


# ---------------------------------------------------------------- 肌力缩放


@case("肌力缩放: F0 槽位实证结果 == gainprm[2] / biasprm[2]")
def test_f0_slots():
    model = _model()
    rep = ma.identify_f0_slots(model)
    assert rep["active_gain_slot"] == ma.F0_GAIN_SLOT, rep
    assert rep["passive_bias_slot"] == ma.F0_BIAS_SLOT, rep


@case("肌力缩放: 1.0 精确还原基准；0.5 → 0.75 无累乘")
def test_scaling_semantics():
    model = _model()
    m = muscle_groups.build_map(model)
    scaler = ma.StrengthScaler(model, m)
    base_gain = scaler.baseline_gain()
    base_bias = scaler.baseline_bias()

    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=1.0, lower_scale=1.0))
    assert np.array_equal(np.asarray(model.actuator_gainprm), base_gain)
    assert np.array_equal(np.asarray(model.actuator_biasprm), base_bias)

    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, lower_scale=0.5))
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.75, lower_scale=0.75))
    mult = scaler.current_multiplier()
    scaled = np.where(np.abs(mult - 1.0) > 1e-12)[0]
    assert scaled.size == 111, scaled.size
    assert np.allclose(mult[scaled], 0.75, rtol=1e-12, atol=1e-12), np.unique(mult)


@case("肌力缩放: 组外（健侧 + 躯干）参数逐元素不变")
def test_untouched_groups():
    model = _model()
    m = muscle_groups.build_map(model)
    scaler = ma.StrengthScaler(model, m)
    base_gain = scaler.baseline_gain()
    base_bias = scaler.baseline_bias()
    scaler.apply(ma.StrengthSpec(paretic_side="L", upper_scale=0.2, lower_scale=0.3))

    keep = sorted(
        set(range(700))
        - set(m.indices_for("L", "upper"))
        - set(m.indices_for("L", "lower"))
    )
    g = np.asarray(model.actuator_gainprm)[keep]
    b = np.asarray(model.actuator_biasprm)[keep]
    assert np.array_equal(g, base_gain[keep])
    assert np.array_equal(b, base_bias[keep])
    assert len(keep) == 700 - 111


@case("肌力缩放: active_only 不改被动力, active_and_passive 改被动力")
def test_modes():
    import mujoco

    model = _model()
    m = muscle_groups.build_map(model)
    scaler = ma.StrengthScaler(model, m)
    q = np.asarray(model.key_qpos[0]).copy()

    def passive_force():
        d = mujoco.MjData(model)
        d.qpos[:] = q
        d.qvel[:] = 0.0
        d.act[:] = 0.0
        mujoco.mj_forward(model, d)
        return np.asarray(d.actuator_force).copy()

    scaler.reset()
    base = passive_force()
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, mode=ma.MODE_ACTIVE_ONLY))
    ao = passive_force()
    scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5, mode=ma.MODE_ACTIVE_AND_PASSIVE))
    ap = passive_force()
    assert np.array_equal(ao, base), "active_only 不应改变被动力"
    assert not np.allclose(ap, base), "active_and_passive 应改变被动力"


@case("肌力缩放: 未知组键/肌肉名会报错，具体肌肉名可覆盖")
def test_group_scales_validation():
    model = _model()
    m = muscle_groups.build_map(model)
    scaler = ma.StrengthScaler(model, m)
    try:
        scaler.apply(ma.StrengthSpec(paretic_side="R", group_scales={"not_a_group": 0.5}))
    except KeyError:
        pass
    else:
        raise AssertionError("未知组键应报 KeyError")
    applied = scaler.apply(ma.StrengthSpec(paretic_side="R", group_scales={"bflh_r": 0.3}))
    mult = applied.multiplier
    names = [e.name for e in m.entries]
    assert abs(mult[names.index("bflh_r")] - 0.3) < 1e-12
    assert applied.n_scaled == 1


@case("肌力缩放: include_shoulder_girdle 会把 44 块躯干肩胛带肌肉纳入上肢组")
def test_shoulder_girdle_option():
    model = _model()
    m = muscle_groups.build_map(model)
    scaler = ma.StrengthScaler(model, m)
    sg = scaler.shoulder_girdle_indices()
    assert len(sg["R"]) == 22, len(sg["R"])
    assert len(sg["L"]) == 22, len(sg["L"])
    a = scaler.apply(ma.StrengthSpec(paretic_side="R", upper_scale=0.5))
    b = scaler.apply(
        ma.StrengthSpec(paretic_side="R", upper_scale=0.5, include_shoulder_girdle=True)
    )
    assert b.n_scaled == a.n_scaled + 22, (a.n_scaled, b.n_scaled)


# ---------------------------------------------------------------- 终止配置


@case("终止配置: JSON 往返 + 注释键被忽略")
def test_termination_json():
    cfg = TerminationConfig(kind="research", min_pelvis_height=0.5, max_pelvis_up_tilt_deg=45.0)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        cfg.to_json(p)
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["_comment"] = "note"
        p.write_text(json.dumps(payload), encoding="utf-8")
        back = TerminationConfig.from_json(p)
    assert back.min_pelvis_height == 0.5
    assert back.max_pelvis_up_tilt_deg == 45.0


@case("终止配置: 拒绝「因偏离参考姿态而终止」的字段组合")
def test_termination_rejects_reference_termination():
    try:
        TerminationConfig(kind="research", allow_reference_deviation_termination=True)
    except ValueError:
        return
    raise AssertionError("应拒绝 allow_reference_deviation_termination=True")


@case("终止配置: 未知字段报错，非法 kind 报错")
def test_termination_bad_fields():
    for payload in ({"kind": "nope"}, {"min_pelvis_height": 1.0, "whoops": 1}):
        try:
            TerminationConfig.from_dict(payload)
        except ValueError:
            continue
        raise AssertionError(f"应报错: {payload}")


@case("终止配置: 骨盆过低 / 直立偏差过大 / 数值异常 都能触发")
def test_termination_triggers():
    import mujoco

    from hemirl.termination import pelvis_upright_local_axis, upright_tilt_deg

    model = _model()
    pelvis = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    ref = np.asarray(model.key_qpos[0]).copy()
    data = mujoco.MjData(model)
    data.qpos[:] = ref
    mujoco.mj_forward(model, data)
    upright = pelvis_upright_local_axis(model, ref, pelvis)
    z_ref = float(data.xpos[pelvis][2])
    assert z_ref > 0.9, z_ref  # 参考姿态的骨盆高度约 0.95 m

    cfg = TerminationConfig(kind="research", min_pelvis_height=0.55)

    # (1) 参考姿态下不触发任何条件
    r = evaluate_research_termination(model, data, cfg, pelvis, 0.0, upright_local=upright)
    assert r is None, f"参考姿态下不应触发终止，但得到 {r}"

    # (2) 高度条件：把阈值抬到参考高度之上即可验证判据本身
    cfg_high = TerminationConfig(kind="research", min_pelvis_height=z_ref + 0.1)
    r = evaluate_research_termination(model, data, cfg_high, pelvis, 0.0, upright_local=upright)
    assert r is not None and "骨盆高度" in r, r

    # (3) 直立偏差判据：纯函数验证（用绕世界 x 轴旋转 80° 的合成旋转矩阵，
    #     避免把躯体压进地面导致 Newton 求解器报 rank-deficient Hessian）
    R_ref = np.asarray(data.xmat[pelvis]).reshape(3, 3)
    ang = np.radians(80.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    tilt = upright_tilt_deg(R_ref @ Rx, upright)
    assert abs(tilt - 80.0) < 1e-6, tilt

    cfg_tilt = TerminationConfig(
        kind="research", min_pelvis_height=None, max_pelvis_up_tilt_deg=60.0
    )
    data.xmat[pelvis] = (R_ref @ Rx).reshape(-1)
    r = evaluate_research_termination(model, data, cfg_tilt, pelvis, 0.0, upright_local=upright)
    assert r is not None and "直立偏差" in r, (r, tilt)

    # (4) NaN 条件：直接写坏 qpos（不再调用 mj_forward，避免求解器在 NaN 上崩溃）
    data.qpos[:] = ref
    data.qpos[5] = np.nan
    r = evaluate_research_termination(model, data, cfg, pelvis, 0.0, upright_local=upright)
    assert r is not None and "数值异常" in r, r


# ---------------------------------------------------------------- 策略组展开


@case("策略组展开: 138 组 → 700 肌肉，索引语义与 DynSynLayer 一致")
def test_group_expansion():
    groups = [
        [0, 1, 5],
        [2, 3],
        [4],
        *[[i] for i in range(6, 700)],
    ]
    exp = build_group_expansion(groups)
    assert exp["muscle_dims"] == 700
    assert exp["muscle_group_nums"] == len(groups)
    gof = exp["group_of_muscle"]
    assert gof[0] == 0 and gof[1] == 0 and gof[5] == 0
    assert gof[2] == 1 and gof[3] == 1 and gof[4] == 2
    a = np.linspace(-1, 1, len(groups))
    out = expand_group_action(a, gof)
    assert out.shape == (700,)
    assert out[0] == out[1] == out[5] == a[0]
    assert out[2] == out[3] == a[1]
    assert np.abs(out).max() <= 1.0


@case("策略组展开: 真实 checkpoint 的 138 个分组覆盖全部 700 块肌肉")
def test_real_group_expansion():
    from hemirl.policy import load_policy_kwargs

    data = load_policy_kwargs(paths.checkpoint_dir("LocomotionFull"))
    groups = data["policy_kwargs"]["dynsyn"]
    exp = build_group_expansion(groups)
    assert exp["muscle_dims"] == 700
    assert exp["muscle_group_nums"] == 138
    assert len(np.unique(exp["group_of_muscle"])) == 138
    assert int(data["target_entropy"]) == -138


# ---------------------------------------------------------------- 兼容层


@case("numpy 兼容层: 幂等且当前环境可用")
def test_numpy_compat():
    a = install_numpy2_pickle_compat()
    b = install_numpy2_pickle_compat()
    assert a["numpy_version"] == b["numpy_version"]
    assert b["already_installed"] or b["needed"]


@case("官方动作包装器: 与参考 sigmoid 逐点一致")
def test_wrapper_equivalence():
    from hemirl.wrappers import verify_wrapper_equivalence

    rep = verify_wrapper_equivalence()
    assert rep["equivalent"], rep


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    n_ok = sum(1 for r in RESULTS if r["ok"])
    print(f"\n{n_ok}/{len(RESULTS)} 通过")
    out = paths.REPORTS_ROOT / "unit_tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": len(RESULTS), "n_pass": n_ok, "results": RESULTS}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {out}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
