"""训练前后对比：策略 / 奖励 / 价值估计 / 动作 / 归一化。

任务书要求：若训练未改善，必须比较训练前后的这些量并给出证据，而不是把失败说成成功。

本脚本在**同一批状态**上比较 A（原始官方策略）与 B（微调策略），因此变化只来自训练：

1. **动作**：同一归一化观测上 ``a_A`` vs ``a_B`` 的最大/平均绝对差、动作饱和比例；
2. **奖励**：同一条 A 轨迹上逐分量给出「官方奖励」与「微调奖励」的量级对比；
3. **价值估计**：在**同一状态-动作对**上比较 ``Q_A`` 与 ``Q_B``，
   并与折扣蒙特卡洛回报对照，判断 critic 是否随奖励尺度一起移动；
4. **归一化**：核对两个 checkpoint 的 ``VecNormalize`` 统计是否相同（本项目冻结统计，
   因此这里应当逐位一致；不一致说明加载有问题）。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/compare_policies.py \\
        --policies A=artifacts/checkpoints/LocomotionFull B=runs/train/healthy_v1 \\
        --out reports/train_vs_baseline.json
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
from hemirl.research_env import ResearchEnvConfig, build_research_env  # noqa: E402
from hemirl.reward_healthy import HealthyRewardConfig, compute as compute_extra  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402


def load_policy(name: str, d: Path, seconds: float):
    from hemirl import policy as pol

    env, raw = build_research_env(
        env_cfg=ResearchEnvConfig(
            termination=TerminationConfig(kind="research"),
            max_episode_seconds=seconds,
            keep_ledger=False,
            name=f"cmp_{name}",
        ),
        checkpoint_dir=d,
    )
    stack = pol.load_sb3_stack(d, env, deterministic=True)
    return env, raw, stack


def rollout(env, raw, stack, seed: int, n_steps: int, extra_cfg: HealthyRewardConfig,
            w_survival: float) -> Dict[str, np.ndarray]:
    """跑一段轨迹并记录观测/动作/两种奖励的分量。"""
    import mujoco

    from hemirl.rollout import root_state

    obs, _ = env.reset(seed=seed)
    Z, A, Roff, Rnew, comp = [], [], [], [], []
    pid = int(env.pelvis_id)
    model, data = env.raw_env.model, env.raw_env.data

    for _ in range(n_steps):
        z = np.asarray(stack.normalize_obs(obs), dtype=np.float32)
        a = np.asarray(stack.predict(z), dtype=float)
        Z.append(z)
        A.append(a)
        obs, reward, terminated, truncated, info = env.step(a)
        rc = info.get("reward_components") or {}
        rs = root_state(model, data, pid)
        terms, extra_total = compute_extra(
            extra_cfg,
            pelvis_y=float(rs["pos"][1]),
            ref_y=env._ref_y(),
            pelvis_y_start=env._pelvis_y_start,
            lin_vel=np.asarray(rs["lin_vel"], dtype=float),
            slip=env._foot_slip(),
        )
        imitation = float(rc.get("imitation", 0.0))
        energy = float(rc.get("energy", 0.0))
        healthy = float(rc.get("official_healthy", 0.0))
        Roff.append(imitation + energy + healthy)
        Rnew.append(imitation + energy + (w_survival if rc.get("survival_physical", 0.0) else 0.0) + extra_total)
        comp.append({"imitation": imitation, "energy": energy, "official_healthy": healthy,
                     "survival_physical": float(rc.get("survival_physical", 0.0)), **terms,
                     "extra_total": extra_total})
        if terminated or truncated:
            break
    return {"z": np.asarray(Z), "a": np.asarray(A), "r_official": np.asarray(Roff),
            "r_new": np.asarray(Rnew), "components": comp}


def discounted_returns(r: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    out = np.zeros_like(r)
    acc = 0.0
    for i in range(len(r) - 1, -1, -1):
        acc = r[i] + gamma * acc
        out[i] = acc
    return out


def q_values(model, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        zt = torch.as_tensor(np.asarray(z, dtype=np.float32))
        at = torch.as_tensor(np.asarray(a, dtype=np.float32))
        qs = model.critic(zt, at)
        return torch.stack([q.mean(dim=-1) for q in qs], dim=-1).mean(dim=-1).numpy()


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description="训练前后对比")
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--config", type=str, default=str(paths.CONFIGS_ROOT / "train_healthy_v1.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "train_vs_baseline.json"))
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    w_survival = float(cfg["reward_split"]["w_survival"])
    extra_cfg = HealthyRewardConfig(**cfg["extra_reward"])

    policies = []
    for item in args.policies:
        name, _, d = item.partition("=")
        policies.append((name, Path(d)))

    report: Dict[str, Any] = {"entry": "scripts/compare_policies.py", "args": vars(args),
                              "seed": args.seed, "steps": args.steps}
    loaded = {}
    for name, d in policies:
        env, raw, stack = load_policy(name, d, seconds=20.0)
        loaded[name] = (env, raw, stack)
        print(f"已加载 {name}: {d}")

    A_name, B_name = policies[0][0], policies[1][0]
    envA, rawA, stackA = loaded[A_name]
    envB, rawB, stackB = loaded[B_name]

    # ---------- 归一化：两者应逐位一致（本项目冻结统计）
    vnA, vnB = stackA.vec_normalize, stackB.vec_normalize
    report["vec_normalize"] = {
        "A_clip_obs": float(vnA.clip_obs), "B_clip_obs": float(vnB.clip_obs),
        "A_count": float(vnA.obs_rms.count), "B_count": float(vnB.obs_rms.count),
        "mean_max_abs_diff": float(np.max(np.abs(np.asarray(vnA.obs_rms.mean)
                                                 - np.asarray(vnB.obs_rms.mean)))),
        "var_max_abs_diff": float(np.max(np.abs(np.asarray(vnA.obs_rms.var)
                                                - np.asarray(vnB.obs_rms.var)))),
        "identical": bool(np.array_equal(np.asarray(vnA.obs_rms.mean), np.asarray(vnB.obs_rms.mean))
                          and np.array_equal(np.asarray(vnA.obs_rms.var), np.asarray(vnB.obs_rms.var))),
        "note": "本项目训练与评估都冻结官方统计（vec_normalize_update=false），故应完全一致",
    }
    print(f"归一化一致: {report['vec_normalize']['identical']} "
          f"(mean diff {report['vec_normalize']['mean_max_abs_diff']:.3e})")

    # ---------- 同一条 A 轨迹上对比
    rollA = rollout(envA, rawA, stackA, args.seed, args.steps, extra_cfg, w_survival)
    z = rollA["z"]
    aA = rollA["a"]

    # 策略：同一批 z 上 B 的动作
    with torch.no_grad():
        aB = np.asarray(stackB.model.policy.actor(
            torch.as_tensor(z.astype(np.float32)), deterministic=True).numpy())
    da = np.abs(aA - aB)
    report["policy_change"] = {
        "n_states": int(z.shape[0]),
        "action_max_abs_diff": float(da.max()),
        "action_mean_abs_diff": float(da.mean()),
        "per_step_max": [float(v) for v in da.max(axis=1)],
        "frac_components_changed_gt_0.1": float(np.mean(da > 0.1)),
        "saturation_A": float(np.mean(np.abs(aA) > 0.99)),
        "saturation_B": float(np.mean(np.abs(aB) > 0.99)),
    }
    print(f"策略变化: mean|Δa|={da.mean():.4f} max={da.max():.4f} "
          f"饱和 A={report['policy_change']['saturation_A']:.4f} B={report['policy_change']['saturation_B']:.4f}")

    # ---------- 奖励分量量级
    comps = rollA["components"]
    report["reward_components"] = {
        "n_steps": len(comps),
        "mean": {k: float(np.mean([c[k] for c in comps])) for k in comps[0]},
        "official_reward_mean": float(np.mean(rollA["r_official"])),
        "new_reward_mean": float(np.mean(rollA["r_new"])),
        "official_reward_sum": float(np.sum(rollA["r_official"])),
        "new_reward_sum": float(np.sum(rollA["r_new"])),
    }
    print("奖励分量均值:", json.dumps({k: round(v, 3) for k, v in
                                  report["reward_components"]["mean"].items()}, ensure_ascii=False))

    # ---------- 价值估计：同一状态-动作对上的 Q
    qA = q_values(stackA.model, z, aA)
    qB = q_values(stackB.model, z, aA)
    G_off = discounted_returns(rollA["r_official"])
    G_new = discounted_returns(rollA["r_new"])
    k = min(20, len(z))
    report["value_estimates"] = {
        "n_states": int(len(z)),
        "Q_A_mean": float(qA.mean()), "Q_B_mean": float(qB.mean()),
        "Q_A_first10": [float(v) for v in qA[:k]], "Q_B_first10": [float(v) for v in qB[:k]],
        "delta_Q_mean": float((qB - qA).mean()),
        "delta_Q_max_abs": float(np.abs(qB - qA).max()),
        "MC_return_official_t0": float(G_off[0]),
        "MC_return_new_t0": float(G_new[0]),
        "Q_vs_MC_official_t0_ratio": float(qA[0] / G_off[0]) if abs(G_off[0]) > 1e-9 else None,
        "Q_vs_MC_new_t0_ratio": float(qB[0] / G_new[0]) if abs(G_new[0]) > 1e-9 else None,
        "note": ("Q 在同一状态-动作对 (z, a_A) 上比较，因此差异只来自 critic 的变化；"
                 "MC 回报由同一条 A 轨迹的两种奖励定义各自算出"),
    }
    print(f"价值: Q_A_mean={qA.mean():.1f} Q_B_mean={qB.mean():.1f} "
          f"ΔQ_max|.|={np.abs(qB-qA).max():.1f}")
    print(f"  MC 回报 t0: official={G_off[0]:.1f} new={G_new[0]:.1f}")

    # ---------- 结论
    improved_alive = float(np.mean([1.0]))  # 由配对评估提供，这里只做数值对比
    report["conclusion"] = {
        "policy_changed": bool(da.mean() > 0.01),
        "value_function_moved": bool(abs(float((qB - qA).mean())) > 1e-6),
        "normalization_unchanged": bool(report["vec_normalize"]["identical"]),
        "reward_scale_ratio_new_over_official": float(
            report["reward_components"]["new_reward_mean"]
            / max(1e-9, abs(report["reward_components"]["official_reward_mean"]))
        ),
    }
    for env, _r, _s in loaded.values():
        env.close()

    provenance.write_json(Path(args.out), report)
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
