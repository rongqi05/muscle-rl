"""针对长时失稳的**少量命名对照实验**——每次只改一个因素。

诊断（``reports/instability_diag.json``）给出的可疑因素只有三个，因此只做三个对照：

**E1 观测裁剪**（``clip_obs`` 10 → 1e9）
    诊断发现：状态类观测分组的裁剪前标准化值 ``z_raw`` 最大到 17.8（阈值 10），
    而**参考类分组从不被裁剪**（``z_raw`` ≤ 3.2）。因此「策略看不到完整状态」是一个
    有证据的可疑因素。做法：同一策略、同一配对种子，只把 ``clip_obs`` 放开，看失稳是否改变。
    **注意**：这会改变策略输入分布（它只在裁剪到 10 的分布上训练过），所以本实验的目的是
    判断「裁剪是不是主因」，而不是找一个更好的评估设置。

**E2 裁剪造成的动作改变量**（纯前向，不改环境）
    对同一条轨迹的每一步，比较 ``actor(clip(z_raw))`` 与 ``actor(z_raw)``。
    这是「信息损失有多大」的直接量化，不依赖任何 rollout 结果的解释。

**E3 参考超前一个控制步**（纯前向，不改环境）
    上游用 ``ref_time = data.time + init_time + dt``，即参考**超前当前状态 1 个控制步**。
    由于参考块整体平移，第 k 步的「无超前」参考块**恰好等于第 k−1 步的参考块**，
    因此可以精确构造单因素对照：只替换观测里的参考切片，其余不动，再比较动作。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/named_controls.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402
from hemirl.instability_diag import obs_group_slices  # noqa: E402


def rollout_with_obs(env, stack, seed: int, max_steps: int) -> Dict[str, Any]:
    """跑一条轨迹并**保留逐步的原始观测**（E2/E3 需要）。"""
    obs, _ = env.reset(seed=seed)
    obs_hist: List[np.ndarray] = []
    src = None
    while True:
        obs_hist.append(np.asarray(obs, dtype=np.float32).copy())
        z = np.asarray(stack.normalize_obs(obs), dtype=np.float32)
        action = stack.predict(z)
        obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            src = info["termination"]["termination_source"]
            break
        if len(obs_hist) >= max_steps:
            src = "local_cap"
            break
    return {"obs": np.asarray(obs_hist), "termination_source": src, "n": len(obs_hist)}


def main() -> int:
    import torch

    from hemirl import policy as pol
    from hemirl.envs import OfficialEnvConfig, build_env
    from hemirl.research_env import ResearchEnvConfig, build_research_env
    from hemirl.termination import TerminationConfig

    parser = argparse.ArgumentParser(description="长时失稳的命名对照实验")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "named_controls.json"))
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    report: Dict[str, Any] = {"entry": "scripts/named_controls.py", "args": vars(args),
                              "factors_changed": {
                                  "E1": ["观测裁剪阈值 clip_obs（唯一改动）"],
                                  "E2": ["无（纯前向测量）"],
                                  "E3": ["观测中的参考块（唯一改动）"],
                              }}
    slices = obs_group_slices()
    ref_slices = [slices["qpos_ref"], slices["qpos_ref_future"]]

    term = TerminationConfig(kind="research")

    # ---------------- E1: clip_obs 放开 ----------------
    print("=== E1 观测裁剪对照（clip_obs 10 vs 1e9，固定策略，配对种子）===")
    e1: Dict[str, Any] = {}
    for label, clip in (("clip10", 10.0), ("noclip", 1.0e9)):
        env_cfg = OfficialEnvConfig.from_checkpoint(ckpt)
        env, raw = build_research_env(
            env_config=env_cfg,
            env_cfg=ResearchEnvConfig(termination=term, max_episode_seconds=args.episode_seconds,
                                      keep_ledger=False, name=label),
        )
        stack = pol.load_sb3_stack(ckpt, env, deterministic=True)
        stack.vec_normalize.clip_obs = clip
        rows = []
        for seed in args.seeds:
            obs, _ = env.reset(seed=seed)
            steps = 0
            while True:
                z = np.asarray(stack.normalize_obs(obs), dtype=np.float32)
                a = stack.predict(z)
                obs, _r, term_, trunc_, info = env.step(a)
                steps += 1
                if term_ or trunc_:
                    break
            rows.append({"seed": seed, "steps": steps,
                         "alive_time_s": info["termination"]["elapsed_time_s"],
                         "termination_source": info["termination"]["termination_source"],
                         "termination_reason": info["termination"]["termination_reason"]})
        env.close()
        e1[label] = {
            "clip_obs": clip,
            "episodes": rows,
            "mean_alive_time_s": float(np.mean([r["alive_time_s"] for r in rows])),
            "n_fell": int(sum(1 for r in rows if r["termination_source"] == "physical_fall")),
        }
        print(f"  {label:8s} clip_obs={clip:<10} 平均存活={e1[label]['mean_alive_time_s']:.2f}s "
              f"跌倒={e1[label]['n_fell']}/{len(rows)}")
    e1["paired_delta_alive_time_s"] = float(
        e1["noclip"]["mean_alive_time_s"] - e1["clip10"]["mean_alive_time_s"]
    )
    e1["conclusion"] = (
        "若 noclip 的存活时间显著更长 → 裁剪是原因之一；若几乎不变 → 裁剪是失稳的**结果**而非原因"
    )
    report["E1_observation_clipping"] = e1

    # ---------------- E2/E3: 纯前向测量 ----------------
    print("\n=== E2/E3 纯前向测量（同一条 20 s 轨迹，仅改一个因素）===")
    env, raw = build_research_env(
        env_config=OfficialEnvConfig.from_checkpoint(ckpt),
        env_cfg=ResearchEnvConfig(termination=term, max_episode_seconds=args.episode_seconds,
                                  keep_ledger=False, name="fwd"),
    )
    stack = pol.load_sb3_stack(ckpt, env, deterministic=True)
    vn = stack.vec_normalize
    actor = stack.model.policy.actor
    mean = np.asarray(vn.obs_rms.mean, dtype=np.float64)
    var = np.asarray(vn.obs_rms.var, dtype=np.float64)
    eps = float(getattr(vn, "epsilon", 1e-8))

    e2_all: List[Dict[str, Any]] = []
    e3_all: List[Dict[str, Any]] = []
    for seed in args.seeds:
        r = rollout_with_obs(env, stack, seed, max_steps=int(args.episode_seconds / env.dt))
        obs_hist = r["obs"]
        n = obs_hist.shape[0]
        z_raw = (obs_hist.astype(np.float64) - mean) / np.sqrt(var + eps)
        z_clip = np.clip(z_raw, -float(vn.clip_obs), float(vn.clip_obs))

        def act(arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(arr, dtype=np.float32))
                return actor(t, deterministic=True).numpy()

        a_clip = act(z_clip)
        a_raw = act(z_raw)
        d2 = np.abs(a_clip - a_raw)
        # 参考块替换：第 k 步的「无超前」参考块 == 第 k-1 步的参考块
        obs_lead0 = obs_hist.copy()
        for sl in ref_slices:
            obs_lead0[1:, sl] = obs_hist[:-1, sl]
        z_lead0 = np.clip((obs_lead0.astype(np.float64) - mean) / np.sqrt(var + eps),
                          -float(vn.clip_obs), float(vn.clip_obs))
        a_lead0 = act(z_lead0)
        d3 = np.abs(a_clip[1:] - a_lead0[1:])

        e2_all.append({
            "seed": seed, "n_steps": n,
            "max_abs_action_diff": float(d2.max()),
            "mean_abs_action_diff": float(d2.mean()),
            "frac_steps_diff_gt_0.1": float(np.mean(d2.max(axis=1) > 0.1)),
            "n_components_clipped_mean": float(np.mean(np.sum(np.abs(z_raw) > float(vn.clip_obs), axis=1))),
        })
        e3_all.append({
            "seed": seed, "n_steps": n,
            "max_abs_action_diff": float(d3.max()),
            "mean_abs_action_diff": float(d3.mean()),
            "frac_steps_diff_gt_0.1": float(np.mean(d3.max(axis=1) > 0.1)),
        })
        print(f"  seed={seed}: E2 max|Δa|={e2_all[-1]['max_abs_action_diff']:.4f} "
              f"mean={e2_all[-1]['mean_abs_action_diff']:.4f} "
              f"被裁剪分量均值={e2_all[-1]['n_components_clipped_mean']:.1f}  ||  "
              f"E3 max|Δa|={e3_all[-1]['max_abs_action_diff']:.4f} "
              f"mean={e3_all[-1]['mean_abs_action_diff']:.4f}")

    env.close()
    report["E2_clipping_effect_on_action"] = {
        "method": "同一轨迹逐步比较 actor(clip(z_raw)) 与 actor(z_raw)，纯前向，不改环境",
        "episodes": e2_all,
        "mean_max_abs_action_diff": float(np.mean([e["max_abs_action_diff"] for e in e2_all])),
        "mean_mean_abs_action_diff": float(np.mean([e["mean_abs_action_diff"] for e in e2_all])),
    }
    report["E3_reference_lead_effect_on_action"] = {
        "method": ("把观测里的参考块换成前一步的参考块（等价于去掉上游 ref_time 的 +dt 超前），"
                   "其余观测完全不动，再比较动作"),
        "episodes": e3_all,
        "mean_max_abs_action_diff": float(np.mean([e["max_abs_action_diff"] for e in e3_all])),
        "mean_mean_abs_action_diff": float(np.mean([e["mean_abs_action_diff"] for e in e3_all])),
    }
    report["provenance"] = provenance.run_provenance(
        entry="scripts/named_controls.py", args=vars(args), checkpoint_dir=ckpt
    )
    provenance.write_json(Path(args.out), report)
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
