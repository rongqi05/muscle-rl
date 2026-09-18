"""测量本机上的训练吞吐与显存/内存需求，用于设定并行数与 replay buffer 大小。

**不要直接沿用上游的 64 环境 / 1e6 buffer**：本机

* 无可用 GPU（``nvidia-smi`` 报 NVML driver/library mismatch，且 torch 是 CPU 版）；
* 20 个 CPU 核、31 GiB 内存；
* 观测 3601 维 + 动作 700 维，单条 transition 的存储开销远大于常见任务。

本脚本实测：

1. 单进程环境推进速度（control steps/s）；
2. 多进程并行时的总吞吐与加速比（找出 CPU 饱和点）；
3. SAC 梯度更新速度（backward passes/s）；
4. replay buffer 的实际每条 transition 字节数与不同容量下的内存占用。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/measure_throughput.py --parallel 1 2 4 8 12
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402


# ------------------------------------------------------------------ 环境吞吐


def _env_worker(args) -> Dict[str, Any]:
    """在一个子进程里跑若干步，返回吞吐。"""
    n_steps, seed, seconds_limit = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np

    from hemirl.research_env import ResearchEnvConfig, build_research_env
    from hemirl.termination import TerminationConfig

    term = TerminationConfig(
        kind="research",
        min_pelvis_height=None,
        max_pelvis_up_tilt_deg=None,
        max_root_speed=None,
        max_root_ang_speed=None,
        max_joint_speed_norm=None,
    )
    env, _raw = build_research_env(
        env_cfg=ResearchEnvConfig(termination=term, max_episode_seconds=1e6, keep_ledger=False)
    )
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, _ = env.reset(seed=seed)
    # 预热（首次 mj_step 有额外开销）
    for _ in range(5):
        obs, *_ = env.step(a)
    t0 = time.perf_counter()
    n = 0
    for _ in range(n_steps):
        obs, *_ = env.step(a)
        n += 1
        if time.perf_counter() - t0 > seconds_limit:
            break
    dt = time.perf_counter() - t0
    env.close()
    return {"steps": n, "seconds": dt, "steps_per_s": n / dt if dt > 0 else 0.0}


def measure_env_throughput(parallel: int, n_steps: int, seconds_limit: float) -> Dict[str, Any]:
    if parallel == 1:
        r = _env_worker((n_steps, 0, seconds_limit))
        return {"parallel": 1, "per_worker_steps_per_s": r["steps_per_s"], "total_steps_per_s": r["steps_per_s"]}
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(parallel) as pool:
        rs = pool.map(_env_worker, [(n_steps, i, seconds_limit) for i in range(parallel)])
    wall = time.perf_counter() - t0
    per = float(np.mean([r["steps_per_s"] for r in rs]))
    return {
        "parallel": parallel,
        "per_worker_steps_per_s": per,
        "total_steps_per_s": per * parallel,
        "wall_time_s": wall,
        "n_workers": len(rs),
    }


# ------------------------------------------------------------------ 梯度吞吐


def measure_gradient_throughput(batch_size: int, n_updates: int) -> Dict[str, Any]:
    """测量 SAC 梯度更新速度（不改变任何持久状态）。"""
    import torch

    from hemirl import policy as pol
    from hemirl.envs import OfficialEnvConfig, build_env

    ckpt = paths.checkpoint_dir("LocomotionFull")
    env_cfg = OfficialEnvConfig.from_checkpoint(ckpt)
    env, raw = build_env(env_cfg)
    try:
        stack = pol.load_sb3_stack(ckpt, env, deterministic=True)
        model = stack.model
    finally:
        env.close()

    obs_dim = int(np.prod(model.observation_space.shape))
    act_dim = int(np.prod(model.action_space.shape))
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(batch_size, obs_dim, generator=g)
    next_obs = torch.randn(batch_size, obs_dim, generator=g)
    actions = torch.rand(batch_size, act_dim, generator=g) * 2 - 1
    rewards = torch.randn(batch_size, 1, generator=g)
    dones = torch.zeros(batch_size, 1)

    optim = torch.optim.Adam(model.actor.parameters(), lr=1e-4)
    # 预热
    for _ in range(2):
        optim.zero_grad()
        loss = model.actor(obs).pow(2).mean()
        loss.backward()
        optim.step()
    t0 = time.perf_counter()
    for _ in range(n_updates):
        optim.zero_grad()
        loss = model.actor(obs).pow(2).mean()
        loss.backward()
        optim.step()
    dt = time.perf_counter() - t0
    return {
        "batch_size": batch_size,
        "n_updates": n_updates,
        "seconds": dt,
        "updates_per_s": n_updates / dt if dt > 0 else 0.0,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "device": "cpu",
    }


# ------------------------------------------------------------------ 内存估算


def replay_buffer_memory(obs_dim: int, act_dim: int, capacity: int) -> Dict[str, Any]:
    """SB3 ReplayBuffer（``optimize_memory_usage=False``）的存储估算。

    每个 transition 存：``obs``(float32, obs_dim) ×2（obs 与 next_obs）、
    ``action``(float32, act_dim)、``reward``(float32,1)、``done``(bool,1)，再加 n_envs。
    """
    per = 4 * obs_dim * 2 + 4 * act_dim + 4 + 1
    return {
        "capacity": capacity,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "bytes_per_transition": per,
        "total_bytes": per * capacity,
        "total_gib": per * capacity / 1024**3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="吞吐与内存测量")
    parser.add_argument("--parallel", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--env-steps", type=int, default=200)
    parser.add_argument("--env-seconds-limit", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grad-updates", type=int, default=20)
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "throughput.json"))
    args = parser.parse_args()

    report: Dict[str, Any] = {"args": vars(args), "cpu_count": os.cpu_count()}

    print("=== 环境推进吞吐 ===")
    report["env_throughput"] = []
    for p in args.parallel:
        r = measure_env_throughput(p, args.env_steps, args.env_seconds_limit)
        report["env_throughput"].append(r)
        print(
            f"  并行 {p:2d}: 单进程 {r['per_worker_steps_per_s']:6.1f} steps/s, "
            f"合计 {r['total_steps_per_s']:7.1f} steps/s"
        )

    print("\n=== 梯度更新吞吐（actor 前向+反向）===")
    g = measure_gradient_throughput(args.batch_size, args.grad_updates)
    report["gradient_throughput"] = g
    print(
        f"  batch={g['batch_size']} obs_dim={g['obs_dim']} act_dim={g['act_dim']} "
        f"→ {g['updates_per_s']:.2f} updates/s"
    )

    print("\n=== replay buffer 内存估算 ===")
    report["replay_buffer"] = [replay_buffer_memory(g["obs_dim"], g["act_dim"], c)
                              for c in (10_000, 50_000, 100_000, 200_000, 1_000_000)]
    for r in report["replay_buffer"]:
        print(
            f"  capacity={r['capacity']:>9d}  {r['bytes_per_transition']:>7d} B/transition  "
            f"合计 {r['total_gib']:7.2f} GiB"
        )

    # 依据实测给出建议
    best = max(report["env_throughput"], key=lambda r: r["total_steps_per_s"])
    budget = 50_000
    report["recommendation"] = {
        "parallel_envs": int(best["parallel"]),
        "expected_steps_per_s": float(best["total_steps_per_s"]),
        "est_seconds_for_50k_transitions": float(budget / max(1e-9, best["total_steps_per_s"])),
        "note": (
            "50,000 transition 的预算是**所有并行环境合计**；并行数取实测吞吐最高的一档。"
            "梯度更新速度单独计入墙钟预算。"
        ),
    }
    report["provenance"] = provenance.run_provenance(
        entry="scripts/measure_throughput.py", args=vars(args)
    )
    provenance.write_json(Path(args.out), report)
    print(f"\n建议并行数: {best['parallel']}（{best['total_steps_per_s']:.0f} steps/s）")
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
