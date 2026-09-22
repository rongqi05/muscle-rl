"""终止 / 截断语义的端到端核查（保护本阶段实验的核心机制）。

## 为什么必须单独查这个

本阶段的关键改动是「侧向偏移超限 → 终止」。它之所以能起作用，靠的是一条链：

    研究环境返回 terminated=True, truncated=False
        ↓ SB3 VecEnv:  info["TimeLimit.truncated"] = truncated and not terminated
        ↓ ReplayBuffer.add:  timeouts[pos] = info.get("TimeLimit.truncated", False)
        ↓ ReplayBuffer._get_samples:  dones = dones * (1 - timeouts)
        ↓ TD 目标：dones=1 → **不** bootstrap；dones=0 → bootstrap

如果这条链在任一处写错（例如把侧向终止报成 ``truncated``），SB3 会**继续 bootstrap**，
即 critic 会认为「往侧向漂走之后还有未来回报」，于是终止**根本不产生代价**，
本阶段的实验就变成无效。这不是推理，必须实测。

## 本脚本查什么

1. 三种结束方式各自的 ``(terminated, truncated)`` 与 ``info["TimeLimit.truncated"]``；
2. 把真实 transition 喂进**真实的** ``ReplayBuffer``，读回 ``dones`` / ``timeouts``，
   计算 SB3 实际使用的 TD 掩码 ``dones * (1 - timeouts)``；
3. DummyVecEnv 与 SubprocVecEnv **两条路径都查**（训练用后者）；
4. 评估不更新归一化统计（逐位比较 ``obs_rms``）。

运行::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/check_termination_semantics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402

CKPT = paths.checkpoint_dir("LocomotionFull")
OUT = paths.REPORTS_ROOT / "termination_semantics.json"


def build_one(max_episode_seconds: float, lateral_limit, min_height=0.55, tilt=60.0):
    """构建单个研究环境（与 train_healthy.make_research_env 完全相同的装配方式）。"""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from hemirl.research_env import RewardSplitConfig, ResearchEnvConfig, build_research_env
    from hemirl.reward_healthy import HealthyRewardConfig
    from hemirl.termination import TerminationConfig

    cfg_json = json.loads(
        (paths.CONFIGS_ROOT / "train_healthy_v2.json").read_text(encoding="utf-8")
    )
    env_cfg = ResearchEnvConfig(
        termination=TerminationConfig(
            kind="research",
            min_pelvis_height=min_height,
            max_pelvis_up_tilt_deg=tilt,
            time_limit_s=None,
            max_lateral_dev_from_ref_m=lateral_limit,
        ),
        max_episode_seconds=max_episode_seconds,
        reward_mode="split",
        reward_split=RewardSplitConfig(
            w_survival=float(cfg_json["reward_split"]["w_survival"]),
            include_official_healthy=bool(cfg_json["reward_split"]["include_official_healthy"]),
        ),
        extra_reward=HealthyRewardConfig(**cfg_json["extra_reward"]),
        keep_ledger=False,
        name="semantics-check",
    )
    env, _raw = build_research_env(env_cfg=env_cfg, checkpoint_dir=CKPT)
    return env


def probe_episode(env, seed: int, policy_action, max_steps: int = 1500):
    """跑一个回合直到结束，返回最后一次 step 的 (terminated, truncated, info)。"""
    obs, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        a = policy_action(obs)
        obs, _r, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            return terminated, truncated, info
    raise AssertionError("回合未结束")


def official_policy_action_factory(env, vec_normalize):
    """用官方 checkpoint 的策略给出动作（与训练时同一条路径）。"""
    import functools

    from hemirl import policy as pol

    pol.install_numpy2_pickle_compat()
    paths.ensure_msgym_on_path()
    paths.ensure_dynsyn_scripts_on_path()
    paths.ensure_dynsyn_package_on_path()
    from DynSyn import SAC_DynSyn  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([lambda: env])
    model_path = CKPT / "checkpoint" / "best_model.zip"
    with pol.patched_float_schedule():
        model = SAC_DynSyn.load(str(model_path), env=venv, device="cpu", print_system_info=False)

    def act(obs):
        z = vec_normalize.normalize_obs(np.asarray(obs, dtype=np.float32))
        return model.predict(z, deterministic=True)[0]

    return act, model, venv


def zero_action(_obs):
    import numpy as _np

    return _np.zeros(700, dtype=_np.float32)


def replay_mask(venv_like, transition, label):
    """把一条真实 transition 喂进真实 ReplayBuffer，返回 SB3 实际用的 TD 掩码。"""
    from stable_baselines3.common.buffers import ReplayBuffer

    obs_space = venv_like.observation_space
    act_space = venv_like.action_space
    buf = ReplayBuffer(
        buffer_size=4,
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        n_envs=1,
        handle_timeout_termination=True,
    )
    obs, next_obs, action, reward, done, info = transition
    buf.add(
        np.asarray([obs], dtype=np.float32),
        np.asarray([next_obs], dtype=np.float32),
        np.asarray([action], dtype=np.float32),
        np.asarray([reward], dtype=np.float64),
        np.asarray([done], dtype=bool),
        [info],
    )
    stored_timeout = float(buf.timeouts[0, 0])
    stored_done = float(buf.dones[0, 0])
    mask = stored_done * (1.0 - stored_timeout)
    return {
        "label": label,
        "buffer_done": stored_done,
        "buffer_timeout": stored_timeout,
        "td_terminal_mask": mask,
        "bootstraps_from_next_state": bool(mask == 0.0),
    }


def main() -> int:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    vn_src = CKPT / "checkpoint" / "best_env.zip"
    results: dict = {"checks": {}, "rows": []}

    # ---------------------------------------------------------------- 归一化冻结
    env_probe = build_one(6.0, 0.25)
    venv = DummyVecEnv([lambda: env_probe])
    vn = VecNormalize.load(str(vn_src), venv)
    vn.training = False
    vn.norm_reward = False
    mean_before = vn.obs_rms.mean.copy()
    var_before = vn.obs_rms.var.copy()
    count_before = float(vn.obs_rms.count)

    act, model, _venv2 = official_policy_action_factory(env_probe, vn)

    # ------------------------------------------------- 1) 侧向终止（核心机制）
    lat_env = build_one(6.0, lateral_limit=0.25)
    lat_venv = DummyVecEnv([lambda: lat_env])
    lat_act, _m, _v = official_policy_action_factory(lat_env, vn)
    obs = lat_venv.reset()
    lat_row = None
    for _ in range(1500):
        a = lat_act(obs[0])[None, :]
        prev_obs = obs.copy()
        obs, rew, done, infos = lat_venv.step(a)
        info = infos[0]
        if bool(done[0]):
            src = info.get("termination", {}).get("termination_source")
            lat_row = {
                "termination_source": src,
                "termination_reason": info.get("termination", {}).get("termination_reason"),
                "sb3_done": bool(done[0]),
                "info_TimeLimit_truncated": bool(info.get("TimeLimit.truncated", False)),
                "transition": (
                    prev_obs[0],
                    info.get("terminal_observation", obs[0]),
                    a[0],
                    float(rew[0]),
                    bool(done[0]),
                    info,
                ),
            }
            break
    assert lat_row is not None, "侧向终止在 6 s 回合内未触发（检查限值或策略）"
    assert lat_row["termination_source"] == "task_lateral_drift", lat_row["termination_source"]

    mask_lat = replay_mask(lat_venv, lat_row["transition"], "lateral_drift(terminated)")
    results["rows"].append(mask_lat)

    # ------------------------------------------------- 2) 时间截断（应 bootstrap）
    t_env = build_one(0.4, lateral_limit=None)   # 0.4 s 必到时间上限
    t_venv = DummyVecEnv([lambda: t_env])
    t_act, _m2, _v2 = official_policy_action_factory(t_env, vn)
    obs = t_venv.reset()
    t_row = None
    for _ in range(200):
        a = t_act(obs[0])[None, :]
        prev_obs = obs.copy()
        obs, rew, done, infos = t_venv.step(a)
        info = infos[0]
        if bool(done[0]):
            t_row = {
                "termination_source": info.get("termination", {}).get("termination_source"),
                "info_TimeLimit_truncated": bool(info.get("TimeLimit.truncated", False)),
                "transition": (
                    prev_obs[0],
                    info.get("terminal_observation", obs[0]),
                    a[0],
                    float(rew[0]),
                    bool(done[0]),
                    info,
                ),
            }
            break
    assert t_row is not None, "0.4 s 回合未在 200 步内结束"
    assert t_row["termination_source"] == "time_limit", t_row["termination_source"]
    mask_time = replay_mask(t_venv, t_row["transition"], "time_limit(truncated)")
    results["rows"].append(mask_time)

    # ------------------------------------------------- 3) 物理跌倒（应不 bootstrap）
    f_env = build_one(6.0, lateral_limit=None)
    f_venv = DummyVecEnv([lambda: f_env])
    obs = f_venv.reset()
    f_row = None
    for _ in range(1500):
        a = zero_action(obs[0])[None, :]      # 零激活 → 必倒
        prev_obs = obs.copy()
        obs, rew, done, infos = f_venv.step(a)
        info = infos[0]
        if bool(done[0]):
            f_row = {
                "termination_source": info.get("termination", {}).get("termination_source"),
                "info_TimeLimit_truncated": bool(info.get("TimeLimit.truncated", False)),
                "transition": (
                    prev_obs[0],
                    info.get("terminal_observation", obs[0]),
                    a[0],
                    float(rew[0]),
                    bool(done[0]),
                    info,
                ),
            }
            break
    assert f_row is not None, "零激活下 1500 步内未终止"
    mask_fall = replay_mask(f_venv, f_row["transition"], f'"fall:{f_row["termination_source"]}"')
    results["rows"].append(mask_fall)

    # --------------------------------- 4) SubprocVecEnv 路径（训练实际使用的）
    import functools
    import os as _os

    _os.environ.setdefault("MUJOCO_GL", "egl")

    def _factory():
        return build_one(6.0, lateral_limit=0.25)

    sub = SubprocVecEnv([_factory])
    obs = sub.reset()
    sub_row = None
    for _ in range(1500):
        a = lat_act(obs[0])[None, :]
        obs, rew, done, infos = sub.step(a)
        info = infos[0]
        if bool(done[0]):
            sub_row = {
                "termination_source": info.get("termination", {}).get("termination_source"),
                "info_TimeLimit_truncated": bool(info.get("TimeLimit.truncated", False)),
            }
            break
    assert sub_row is not None, "SubprocVecEnv 路径未触发侧向终止"
    results["checks"]["subproc_lateral"] = sub_row
    sub.close()

    # ---------------------------------------------------------------- 断言与汇总
    checks = {
        "lateral_terminated_not_truncated": {
            "pass": lat_row["sb3_done"] and not lat_row["info_TimeLimit_truncated"],
            "sb3_done": lat_row["sb3_done"],
            "TimeLimit.truncated": lat_row["info_TimeLimit_truncated"],
        },
        "lateral_no_bootstrap": {
            "pass": not mask_lat["bootstraps_from_next_state"],
            **mask_lat,
        },
        "time_limit_bootstraps": {
            "pass": mask_time["bootstraps_from_next_state"]
            and bool(t_row["info_TimeLimit_truncated"]),
            **mask_time,
        },
        "physical_fall_no_bootstrap": {
            "pass": not mask_fall["bootstraps_from_next_state"],
            **mask_fall,
        },
        "subproc_same_semantics": {
            "pass": sub_row["termination_source"] == "task_lateral_drift"
            and not sub_row["info_TimeLimit_truncated"],
            **sub_row,
        },
        "eval_does_not_update_normalization": {
            "pass": bool(
                np.array_equal(mean_before, vn.obs_rms.mean)
                and np.array_equal(var_before, vn.obs_rms.var)
                and count_before == float(vn.obs_rms.count)
            ),
            "count_before": count_before,
            "count_after": float(vn.obs_rms.count),
        },
    }
    results["checks"] = checks
    # 评估期间确实调用过 normalize_obs，统计仍未变 → 证明评估只读
    _ = vn.normalize_obs(np.zeros(3601, dtype=np.float32))
    checks["eval_does_not_update_normalization"]["count_after_normalize_call"] = float(
        vn.obs_rms.count
    )
    checks["eval_does_not_update_normalization"]["pass"] = bool(
        np.array_equal(mean_before, vn.obs_rms.mean)
        and np.array_equal(var_before, vn.obs_rms.var)
        and count_before == float(vn.obs_rms.count)
    )

    results["n_pass"] = sum(1 for c in checks.values() if c["pass"])
    results["n_total"] = len(checks)
    results["provenance"] = provenance.run_provenance(
        entry="scripts/check_termination_semantics.py",
        args={},
        extra={"checkpoint_dir": str(CKPT)},
        checkpoint_dir=CKPT,
    )

    print("\n=== 终止/截断语义核查 ===")
    for k, c in checks.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {k}")
    for r in results["rows"]:
        print(
            f"    {r['label']:34s} buffer_done={r['buffer_done']:.0f} "
            f"timeout={r['buffer_timeout']:.0f} TD掩码={r['td_terminal_mask']:.0f} "
            f"bootstrap={r['bootstraps_from_next_state']}"
        )
    print(f"\n{results['n_pass']}/{results['n_total']} 通过")

    provenance.write_json(OUT, results)
    print(f"[saved] {OUT}")
    return 0 if results["n_pass"] == results["n_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
