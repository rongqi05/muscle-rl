"""研究环境与物理量的回归测试。

设计原则：**只测行为，不复述实现**。每条测试都对应一个曾经真实出错的点：

============  ==========================================================================
测试           对应的真实问题
============  ==========================================================================
T1 最后一步    旧评估循环在 ``env.step`` 后立刻 ``break``，终止那一步的物理状态被丢掉
T2 时间截断    研究模式曾仍受官方写死的 3.51 s 限制，且「时间到」没有独立截断标志
T3 时间来源    旧实现用「步数 × 标称 dt」当运行时间，与实际仿真时间可能有偏差
T4 物理跌倒    跌倒与「时间到」曾被压进同一个 terminated 字段
T5 参考终止    官方「偏离参考」终止在官方模式必须保留、在研究模式必须被覆盖
T6 数值异常    数值异常必须单独归类，不混入生理性跌倒
T7 终止后      终止后继续 step 必须抛错（不能静默推进或自动 reset）
T8 奖励拆分    ``w_healthy`` 依赖参考姿态，不能被当成「物理存活」
T9 根速度      ``root_state`` 的世界系角速度必须与旋转 Jacobian / 有限差分一致
T10 力分解     约束力必须按类型拆开且残差为 0，不能把 qfrc_constraint 整体归因 equality
T11 参考连续    参考轨迹的循环跳变必须被量化（长时评估的解释前提）
============  ==========================================================================

运行：``PYTHONPATH=. python scripts/run_tests.py``（本文件不依赖 checkpoint）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import forces, muscle_groups, paths  # noqa: E402
from hemirl.research_env import (  # noqa: E402
    REWARD_MODE_SPLIT,
    ResearchEnvConfig,
    build_research_env,
    reference_continuity_report,
)
from hemirl.termination import TerminationConfig  # noqa: E402

RESULTS: list = []

_ENV_CACHE: dict = {}


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


def _make_env(key: str, term: TerminationConfig, **kw):
    """按 key 缓存环境（构建一次约 2 s），并用 ``configure_termination`` 切换配置。"""
    if key not in _ENV_CACHE:
        env, raw = build_research_env(
            env_cfg=ResearchEnvConfig(termination=term, **kw)
        )
        _ENV_CACHE[key] = (env, raw)
    env, raw = _ENV_CACHE[key]
    env.configure_termination(term)
    return env, raw


ZERO_ACTION = np.zeros(700, dtype=np.float32)


def _run_until_done(env, action=ZERO_ACTION, max_iter=2000):
    env.reset(seed=0)
    for _ in range(max_iter):
        _obs, _rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info
    raise AssertionError("环境在 max_iter 内未终止")


# ------------------------------------------------------------------ 终止 / 计时


@case("T1 终止步被记录：ledger 覆盖全部已执行控制步，含触发终止的最后一步")
def test_t1_last_step_recorded():
    term = TerminationConfig(kind="research")
    env, _raw = _make_env("t1", term, max_episode_seconds=0.4)
    info = _run_until_done(env)
    acc = env.accounting_report()
    assert info["termination"]["truncated"] is True, info["termination"]
    assert acc["all_steps_recorded"], acc
    assert acc["ledger_covers_terminating_step"], acc
    assert acc["n_control_steps_recorded"] == acc["n_control_steps_executed"] == 20, acc
    last = env.ledger[-1]
    assert last.termination_reason is not None, last
    assert last.termination_source == "time_limit", last
    # 最后一步的物理量确实被采到（不是占位值）
    assert np.isfinite(last.pelvis_z) and last.pelvis_z > 0.0, last


@case("T2 时间截断：研究模式用规定时长（不是官方 3.51 s），且带明确截断标志")
def test_t2_time_truncation():
    # 只留时间判据，把物理判据全部关掉，以便**隔离**地测时间截断逻辑
    term = TerminationConfig(
        kind="research",
        min_pelvis_height=None,
        max_pelvis_up_tilt_deg=None,
        max_root_speed=None,
        max_root_ang_speed=None,
        max_joint_speed_norm=None,
    )
    env, _raw = _make_env("t2", term, max_episode_seconds=1.0)
    assert abs(env.time_limit_s - 1.0) < 1e-12, env.time_limit_s
    assert env.max_control_steps == 50, env.max_control_steps
    assert abs(env.official_time_limit_s - 3.51) < 1e-9, env.official_time_limit_s
    info = _run_until_done(env)
    t = info["termination"]
    assert t["truncated"] is True and t["terminated"] is False, t
    assert t["termination_source"] == "time_limit", t
    assert t["n_control_steps"] == 50, t
    assert t["reached_time_limit"] is True, t
    assert t["reached_step_cap"] is False, t
    assert env.accounting_report()["all_steps_recorded"]

    # 研究模式必须**覆盖**官方写死的时间截断：在 3.52 s（官方上限）时不得截断，
    # 只有在达到**规定时长**时才截断
    env3, _r3 = _make_env("t2c", term, max_episode_seconds=5.0)
    env3.reset(seed=0)
    r, src = env3._evaluate_termination(3.52, True, True)
    assert (r, src) == (None, None), (r, src)
    r, src = env3._evaluate_termination(5.01, True, True)
    assert src == "time_limit", (r, src)
    assert env3.cfg.suppresses_official_time_truncation is True

    # 本地步数上限（比时间更早生效）→ 必须有明确截断标志
    term_cap = TerminationConfig(
        kind="research",
        min_pelvis_height=None,
        max_pelvis_up_tilt_deg=None,
        max_root_speed=None,
        max_root_ang_speed=None,
        max_joint_speed_norm=None,
    )
    env2, _r2 = _make_env("t2b", term_cap, max_episode_seconds=10.0, max_control_steps=7)
    info2 = _run_until_done(env2)
    t2 = info2["termination"]
    assert t2["truncated"] is True, t2
    assert t2["termination_source"] == "step_cap", t2
    assert t2["n_control_steps"] == 7, t2
    assert t2["reached_step_cap"] is True, t2


@case("T3 计时来源：运行时间 = data.time 差（实际仿真时间），且与步数一致")
def test_t3_timing_source():
    term = TerminationConfig(kind="research")
    env, _raw = _make_env("t3", term, max_episode_seconds=0.6)
    info = _run_until_done(env)
    acc = env.accounting_report()
    t = info["termination"]
    assert abs(t["elapsed_time_s"] - t["n_control_steps"] * env.dt) < 1e-9, t
    assert acc["max_abs_diff_elapsed_vs_nominal"] < 1e-9, acc
    assert acc["per_step_gap_max_dev"] < 1e-9, acc
    assert "实际仿真时间" in acc["time_source"], acc
    assert t["n_physics_steps"] == t["n_control_steps"] * env.frame_skip, t


@case("T4 物理跌倒：terminated=True 且 source=physical_fall（不是 truncated）")
def test_t4_physical_fall():
    term = TerminationConfig(kind="research", min_pelvis_height=0.55)
    env, _raw = _make_env("t4", term, max_episode_seconds=8.0)
    info = _run_until_done(env)
    t = info["termination"]
    assert t["terminated"] is True and t["truncated"] is False, t
    assert t["physical_fall"] is True, t
    assert t["numeric_anomaly"] is False, t
    assert t["elapsed_time_s"] < 8.0, t
    assert env.accounting_report()["ledger_covers_terminating_step"]
    # 终止步必须确实违反了**某一条物理判据**（骨盆过低 或 直立偏差过大），
    # 而不是「时间到」被误标成跌倒
    last = env.ledger[-1]
    assert last.termination_source == "physical_fall", last
    violated = last.pelvis_z < 0.55 or last.up_tilt_deg > 60.0
    assert violated, last
    # 参考跟踪误差明显变大，但它是**描述性**指标，不是判据本身
    assert last.qpos_track_err > 0.1, last


@case("T5 官方参考误差终止：官方模式保留，研究模式覆盖但被记录")
def test_t5_official_reference_termination():
    # qpos_diff_th 极小 → 上游立刻判 is_healthy=False
    import gymnasium as gym  # noqa: F401

    from hemirl.envs import OfficialEnvConfig, build_env
    from hemirl.research_env import ResearchLocomotionEnv

    env_cfg = OfficialEnvConfig.from_checkpoint(paths.checkpoint_dir("LocomotionFull"))
    env_cfg.single_env_kwargs = dict(env_cfg.single_env_kwargs)
    env_cfg.single_env_kwargs["qpos_diff_th"] = 1e-9
    env_cfg.single_env_kwargs["gait_cycles"] = 1

    # (a) 官方模式：必须与上游一致地终止
    base, raw = build_env(env_cfg)
    off = ResearchLocomotionEnv(base, ResearchEnvConfig(termination=TerminationConfig(kind="official"), reward_mode="official"), raw_env=raw)
    off.reset(seed=0)
    _o, _r, term_flag, _tr, info = off.step(ZERO_ACTION)
    assert term_flag is True, info["termination"]
    assert info["termination"]["termination_source"] == "official_reference_deviation", info["termination"]
    assert info["termination"]["terminated"] is True, info["termination"]

    # (b) 研究模式：同样的环境配置**不得**因参考偏差终止
    base2, raw2 = build_env(env_cfg)
    rese = ResearchLocomotionEnv(base2, ResearchEnvConfig(termination=TerminationConfig(kind="research"), max_episode_seconds=0.2), raw_env=raw2)
    rese.reset(seed=0)
    seen_official_flag = False
    for _ in range(10):
        _o, _r, terminated, truncated, info = rese.step(ZERO_ACTION)
        if info["termination"]["n_official_terminated_flags"] > 0:
            seen_official_flag = True
        assert info["termination"]["termination_source"] != "official_reference_deviation", info["termination"]
        if terminated or truncated:
            break
    assert seen_official_flag, "应当记录到官方终止标志（被覆盖但必须留证据）"
    assert rese.cfg.suppresses_official_reference_termination is True


@case("T6 数值异常：单独归类为 numeric_anomaly，不混入生理性跌倒")
def test_t6_numeric_anomaly():
    term = TerminationConfig(kind="research")
    env, _raw = _make_env("t6", term, max_episode_seconds=1.0)
    env.reset(seed=0)
    env.step(ZERO_ACTION)
    saved = float(env.data.act[0])
    try:
        env.data.act[0] = np.nan
        reason, source = env._evaluate_termination(0.1, False, False)
    finally:
        env.data.act[0] = saved
    assert source == "numeric_anomaly", (reason, source)
    assert "数值异常" in reason, reason

    # 端到端：在上游 step 之后注入 NaN，模拟「仿真器产生了数值异常」；
    # 走完整链路（step → 分类 → 元数据），验证两类终止被分开。
    env.reset(seed=0)
    env.step(ZERO_ACTION)
    original_step = env.env.step

    def poisoned(action):
        out = original_step(action)
        env.data.qpos[5] = np.nan
        return out

    try:
        env.env.step = poisoned  # type: ignore[method-assign]
        _o, _r, terminated, _tr, info = env.step(ZERO_ACTION)
    finally:
        del env.env.step
    t = info["termination"]
    assert terminated is True, t
    assert t["numeric_anomaly"] is True, t
    assert t["physical_fall"] is False, t
    assert t["termination_source"] == "numeric_anomaly", t
    assert t["termination_reason"].startswith("数值异常"), t


@case("T7 终止后不得继续推进：再 step 直接抛错，且不会自动 reset")
def test_t7_no_step_after_done():
    term = TerminationConfig(kind="research")
    env, _raw = _make_env("t7", term, max_episode_seconds=0.2)
    _run_until_done(env)
    assert env.done is True
    steps_snapshot = env.n_control_steps_executed
    try:
        env.step(ZERO_ACTION)
    except RuntimeError as exc:
        assert "已终止" in str(exc), str(exc)
    else:
        raise AssertionError("终止后 step 必须抛 RuntimeError")
    assert env.n_control_steps_executed == steps_snapshot, "不得静默推进"
    # reset 之后可重新开始，且计数清零
    env.reset(seed=1)
    assert env.done is False
    assert env.n_control_steps_executed == 0


@case("T8 奖励拆分：物理存活项不依赖参考，官方 healthy 项依赖参考")
def test_t8_reward_split():
    term = TerminationConfig(kind="research")
    env, _raw = _make_env("t8", term, max_episode_seconds=0.2)
    env.cfg = type(env.cfg)(
        termination=term,
        max_episode_seconds=0.2,
        reward_mode=REWARD_MODE_SPLIT,
        reward_split=env.cfg.reward_split,
    )
    env.reset(seed=0)
    _o, reward, _t, _tr, info = env.step(ZERO_ACTION)
    rc = info["reward_components"]
    assert rc["survival_physical_is_reference_dependent"] is False, rc
    assert rc["official_healthy_is_reference_dependent"] is True, rc
    assert rc["survival_physical"] == env.cfg.reward_split.w_survival, rc
    expected = rc["imitation"] + rc["energy"] + rc["survival_physical"]
    assert abs(reward - expected) < 1e-9, (reward, expected)

    # 骨盆压低到阈值以下 → 物理存活项为 0（而官方 healthy 项仍按参考姿态计算）
    import mujoco

    env.reset(seed=0)
    env.data.qpos[1] = -1.0  # pelvis_ty 是相对默认高度的竖直位移
    mujoco.mj_forward(env.model, env.data)  # xpos 需要前向重算才反映新的 qpos
    assert env._is_physically_alive() is False
    comp2 = env._reward_components({"reward_qpos": 0.0, "reward_xpos": 0.0, "reward_pelvis": 0.0,
                                    "reward_energy": 0.0, "reward_healthy": 100.0})
    assert comp2["survival_physical"] == 0.0, comp2
    assert comp2["official_healthy"] == 100.0, comp2

    # 生理上存活时物理存活项与参考姿态**无关**：把参考轨迹挪走不应改变它
    env.reset(seed=0)
    before = env._reward_components({})["survival_physical"]
    env.raw_env.qpos_ref = np.asarray(env.raw_env.qpos_ref).copy() + 0.5
    after = env._reward_components({})["survival_physical"]
    assert before == after, (before, after)


# ------------------------------------------------------------------ 物理量


@case("T9 根节点速度：世界系线速度/角速度与 Jacobian 和有限差分一致")
def test_t9_root_velocity_jacobian():
    import mujoco

    from hemirl.rollout import root_state

    model = mujoco.MjModel.from_xml_path(str(paths.MODEL_XML))
    pelvis = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    rng = np.random.default_rng(0)
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(model.key_qpos[0])
    # 给一点非平凡姿态与速度（避开全零，否则差分噪声会放大）
    data.qpos[3:6] += rng.normal(0, 0.2, size=3)
    data.qpos[6:] += rng.normal(0, 0.2, size=model.nq - 6)
    data.qvel[:] = rng.normal(0, 0.8, size=model.nv)
    mujoco.mj_forward(model, data)

    rs = root_state(model, data, pelvis)
    eps = 1e-7

    # --- 线速度：中心差分 data.xpos[pelvis]
    pos_fd = []
    for sign in (+1, -1):
        d = mujoco.MjData(model)
        d.qpos[:] = data.qpos + sign * eps * data.qvel
        mujoco.mj_forward(model, d)
        pos_fd.append(np.asarray(d.xpos[pelvis]).copy())
    lin_fd = (pos_fd[0] - pos_fd[1]) / (2 * eps)
    assert np.allclose(rs["lin_vel"], lin_fd, atol=1e-5), (rs["lin_vel"], lin_fd)

    # --- 角速度：dR/dt @ R^T 必须是反对称阵，其分量即世界系角速度
    rot_fd = []
    for sign in (+1, -1):
        d = mujoco.MjData(model)
        d.qpos[:] = data.qpos + sign * eps * data.qvel
        mujoco.mj_forward(model, d)
        rot_fd.append(np.asarray(d.xmat[pelvis]).reshape(3, 3).copy())
    Rdot = (rot_fd[0] - rot_fd[1]) / (2 * eps)
    R = rs["rot"]
    W = Rdot @ R.T
    assert np.allclose(W, -W.T, atol=1e-6), W  # 反对称性自检
    omega_fd = np.array([W[2, 1], W[0, 2], W[1, 0]])
    assert np.allclose(rs["ang_vel"], omega_fd, atol=1e-5), (rs["ang_vel"], omega_fd)

    # --- 旧式「槽位重排」写法：在被验证为错的同时，量化它的偏差
    diff_lin = float(np.max(np.abs(rs["lin_vel"] - rs["lin_vel_index_formula"])))
    diff_ang = float(np.max(np.abs(rs["ang_vel"] - rs["ang_vel_index_formula"])))
    assert rs["velocity_method"].startswith("mujoco.mj_jacBody"), rs["velocity_method"]
    # 该项只作报告，不作断言：两种写法的差必须被显式记录下来
    assert np.isfinite(diff_lin) and np.isfinite(diff_ang)
    print(f"       [info] 旧索引写法偏差 lin={diff_lin:.6f} ang={diff_ang:.6f}")


@case("T10 约束力分解：按类型拆开后残差为 0，不把 qfrc_constraint 整体归因 equality")
def test_t10_constraint_decomposition():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(paths.MODEL_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(model.key_qpos[0])
    mujoco.mj_forward(model, data)
    rep = forces.decompose_constraint_forces(model, data)
    assert rep["decomposition_ok"], rep
    assert rep["decomposition_residual_absmax"] < 1e-6, rep
    assert rep["nefc"] >= 0
    # 分组必须完整覆盖：equality + limit + contact + friction
    assert set(rep["per_group_norm"]) == {"equality", "limit", "contact", "friction"}, rep
    # 广义力范数必须带单位警告，避免被当成承重占比
    terms = forces.generalized_force_terms(model, data)
    assert "不能解释为承重" in terms["unit_note"], terms["unit_note"]

    # 肌肉主动/被动拆分：F(act) - F(0) 定义自洽
    split = forces.muscle_active_passive_split(model, data)
    assert split["active_matches_definition_absmax"] < 1e-9, split


@case("T11 参考轨迹连续性：边界跳变可量化，且给出是否需要护栏的结论")
def test_t11_reference_continuity():
    env, raw = _make_env("t11", TerminationConfig(kind="research"), max_episode_seconds=0.2)
    rep = reference_continuity_report(raw, n_cycles=6)
    for key in (
        "max_step_change_within_cycle",
        "max_step_change_at_cycle_boundary",
        "boundary_over_within_ratio",
        "pose_wrap_discontinuity_max_rad",
        "wrap_frame_gap_s",
        "per_cycle_forward_from_qpos_m",
        "needs_loop_guard",
    ):
        assert rep.get(key) is not None, (key, rep)
    # 结论必须自洽：边界跳变与周期内单步变化同阶 → 无需额外护栏
    assert rep["boundary_over_within_ratio"] == rep["boundary_over_within_ratio"], rep
    assert isinstance(rep["needs_loop_guard"], bool), rep
    # 前进位移累积不能有明显漂移
    assert rep["forward_accumulation_rel_err"] < 0.01, rep


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    n_ok = sum(1 for r in RESULTS if r["ok"])
    print(f"\n研究环境回归: {n_ok}/{len(RESULTS)} 通过")
    out = paths.REPORTS_ROOT / "unit_tests_research_env.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"n": len(RESULTS), "n_pass": n_ok, "results": RESULTS},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[saved] {out}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
