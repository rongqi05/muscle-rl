"""验证策略加载：官方 SB3 路径 vs 纯 torch 直读路径必须一致。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/verify_policy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402


def main() -> None:
    paths.ensure_msgym_on_path()
    paths.ensure_msgym_model_link()
    import gymnasium as gym
    import msgym  # noqa: F401,F811

    from hemirl import policy as pol
    from hemirl.wrappers import official_muscle_norm_wrapper, verify_wrapper_equivalence

    ckpt_dir = paths.checkpoint_dir("LocomotionFull")
    report: dict = {}

    # 1) wrapper 等价性
    rep = verify_wrapper_equivalence()
    report["wrapper_equivalence"] = rep
    print("[wrapper]", rep["equivalent"], rep["sample"])

    # 2) 组展开检查
    data = pol.load_policy_kwargs(ckpt_dir)
    groups = data["policy_kwargs"]["dynsyn"]
    exp = pol.build_group_expansion(groups)
    report["group_expansion"] = {
        "n_groups": exp["muscle_group_nums"],
        "muscle_dims": exp["muscle_dims"],
        "target_entropy_in_ckpt": data.get("target_entropy"),
        "unique_muscle_indices": int(np.unique(exp["group_of_muscle"]).size),
    }
    print("[groups]", report["group_expansion"])

    # 3) 两条加载路径
    env = gym.make(
        "msgym/LocomotionFullEnv-v1",
        skip_frames=10,
        reset_noise_scale=0.001,
        qpos_diff_th=0.06,
        gait_cycles=3,
        random_init=True,
        reward_dict={"w_qpos": 50, "w_xpos": 50, "w_pelvis": 100, "w_energy": 0.1, "w_healthy": 100},
    )
    Wrapper = official_muscle_norm_wrapper()
    env = Wrapper(env)

    stack = pol.load_sb3_stack(ckpt_dir, env, deterministic=True)
    print("[sb3] loaded; policy class:", type(stack.model.policy).__name__)
    print("[sb3] actor class:", type(stack.model.policy.actor).__name__)
    amp = stack.model.policy.actor.dynsyn_layer.dynsyn_weight_amp
    report["dynsyn_weight_amp_at_load"] = amp
    print("[sb3] dynsyn_weight_amp:", amp)

    actor = pol.direct_actor_from_checkpoint(ckpt_dir)
    print("[direct] loaded")

    # 4) 逐点比对：对若干真实观测比较两条路径输出
    diffs = []
    rng = np.random.default_rng(0)
    for i in range(5):
        obs, _ = env.reset(seed=100 + i)
        norm = stack.normalize_obs(obs)
        a_sb3 = stack.predict(norm)
        a_dir = actor(norm, deterministic=True)
        diffs.append(float(np.max(np.abs(a_sb3 - a_dir))))
    report["max_abs_action_diff"] = max(diffs)
    report["per_sample_diffs"] = diffs
    print("[compare] max |a_sb3 - a_direct| =", max(diffs))
    ok = max(diffs) < 1e-5
    print("[compare] 一致:", ok)

    # 5) 动作统计
    obs, _ = env.reset(seed=0)
    norm = stack.normalize_obs(obs)
    a = stack.predict(norm)
    env_action = env.action(np.asarray(a))  # 走 MuscleNormWrapper
    report["action_stats"] = {
        "dim": int(a.shape[-1]),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "env_ctrl_min": float(env_action.min()),
        "env_ctrl_max": float(env_action.max()),
        "n_unique_group_values": int(np.unique(np.round(a, 6)).size),
    }
    print("[action]", report["action_stats"])

    out = paths.REPORTS_ROOT / "verify_policy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out}")
    env.close()


if __name__ == "__main__":
    main()
