"""环境自检：创建官方环境、测量参考轨迹统计、确认动作/观测接口。

用法::

    PYTHONPATH=. python scripts/probe_env.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402


def main() -> None:
    import mujoco

    paths.ensure_msgym_on_path()
    paths.ensure_msgym_model_link()
    import msgym  # noqa: F401,F811
    import gymnasium as gym

    print("registered envs:", [k for k in gym.registry.keys() if "msgym" in k])

    env = gym.make(
        "msgym/LocomotionFullEnv-v1",
        skip_frames=10,
        reset_noise_scale=0.001,
        qpos_diff_th=0.06,
        gait_cycles=3,
        random_init=True,
        reward_dict={"w_qpos": 50, "w_xpos": 50, "w_pelvis": 100, "w_energy": 0.1, "w_healthy": 100},
    )
    u = env.unwrapped
    print("\n=== spaces ===")
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)
    print("dt:", u.dt, " frame_skip:", u.frame_skip, " model timestep:", u.model.opt.timestep)
    print("terminate_time:", u.terminate_time, " cycles:", u.cycles)
    print("num_trajectories:", u.trajectory.num_trajectories)
    print("pelvis_id:", u.pelvis_id, "key_body_ids:", u.key_body_ids)

    props = u.trajectory.get_trajectory_properties(0)
    print("trajectory properties (terminate_time, velocity, stride):", props)

    # 参考轨迹统计
    n = int(u.terminate_time // u.dt)
    times = np.arange(n, dtype=np.float64) * u.dt
    qpos_b, xpos_b, qvel_b = u.trajectory.query_batch(times, 0)
    pelvis_z = xpos_b[:, u.pelvis_id, 2]
    print("\n=== reference pelvis world-z ===")
    print(f"  n_samples={n}  mean={pelvis_z.mean():.4f}  min={pelvis_z.min():.4f}  max={pelvis_z.max():.4f}")

    # 参考轨迹的速度估计（有限差分）
    root_xy = xpos_b[:, u.pelvis_id, :2]
    dxy = np.linalg.norm(np.diff(root_xy, axis=0), axis=1) / u.dt
    print(f"  root horizontal speed: mean={dxy.mean():.4f} m/s  max={dxy.max():.4f} m/s")

    # 姿态：骨盆 up 轴（用参考四元数近似）
    print("\n=== reset 行为 ===")
    obs, info = env.reset(seed=0)
    print("obs shape:", obs.shape, " dtype:", obs.dtype)
    print("qpos[:6] after reset:", np.round(u.data.qpos[:6], 6))
    print("init_time:", u.init_time, " qpos_ref[:6]:", np.round(u.qpos_ref[:6], 6))
    print("terminated:", u.terminated, " is_healthy:", u.is_healthy)

    # 重复 reset 是否会因种子而一致
    obs_a, _ = env.reset(seed=7)
    qa = u.data.qpos.copy()
    obs_b, _ = env.reset(seed=7)
    qb = u.data.qpos.copy()
    print("seed 7 两次 reset 的 qpos 完全一致:", bool(np.array_equal(qa, qb)))

    # 零动作 rollout：确认重力起作用（动力学而非回放）
    print("\n=== 零动作 rollout（前 5 步）===")
    env.reset(seed=0)
    q0 = u.data.qpos.copy()
    z0 = float(u.data.xpos[u.pelvis_id][2])
    for i in range(5):
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, r, term, trunc, inf = env.step(a)
        print(f"  step {i}: pelvis_z={u.data.xpos[u.pelvis_id][2]:.5f}  qvel_norm={np.linalg.norm(u.data.qvel):.4f}  term={term} trunc={trunc}")
    print("  z0:", z0)

    # 关键结构
    m = u.model
    print("\n=== model summary ===")
    print(f"  nq={m.nq} nv={m.nv} nu={m.nu} na={m.na} neq={m.neq} nbody={m.nbody}")
    print("  qfrc_applied (max abs):", float(np.max(np.abs(u.data.qfrc_applied))))
    print("  xfrc_applied (max abs):", float(np.max(np.abs(u.data.xfrc_applied))))
    env.close()
    print("\n[done]")


if __name__ == "__main__":
    main()
