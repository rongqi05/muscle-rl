"""核对「Q vs MC 回报」这一组数字是否自洽（此前报告存在不自洽）。

## 为什么要单独查

上一阶段的报告里写了 $Q/\\mathrm{MC}$ 比值（官方 0.989、微调 1.376），并据此把
「价值函数未收敛」定为下一步的单项调整依据。但把同一份 JSON 里两个数字对在一起：

* `reward_components.official_reward_sum` = **−6341.6**（223 步）
* `value_estimates.MC_return_official_t0` = **+4274.1**

逐步奖励全为负、总和为负，折扣回报 $G_0=\\sum_i\\gamma^i r_i$ 不可能为正。
所以其中至少有一个是错的，**基于它的结论不能继续沿用**。

## 本脚本做什么

1. 用**两种互相独立**的方法算 $G_0$（递推 / 直接加权求和），要求逐位一致；
2. 同时校验 $\\sum_t (1-\\gamma)\\gamma^t G_t \\approx \\sum_t \\gamma^t r_t$（回报的
   一致性恒等式），排除「数组内容与索引错位」；
3. 打印前若干步的原始奖励，便于看清量级。

运行::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/check_value_consistency.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402
from hemirl.reward_healthy import HealthyRewardConfig, compute as compute_extra  # noqa: E402
from hemirl.rollout import root_state  # noqa: E402

GAMMA = 0.99
OUT = paths.REPORTS_ROOT / "value_consistency.json"


def mc_return_recursive(r: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """从后往前的递推（与 compare_policies.discounted_returns 相同算法）。"""
    out = np.zeros_like(r, dtype=float)
    acc = 0.0
    for i in range(len(r) - 1, -1, -1):
        acc = float(r[i]) + gamma * acc
        out[i] = acc
    return out


def mc_return_direct(r: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """独立实现：显式的 $\\sum_i \\gamma^i r_{t+i}$（不用递推）。"""
    n = len(r)
    pw = gamma ** np.arange(n)
    out = np.zeros(n, dtype=float)
    for t in range(n):
        out[t] = float(np.dot(pw[: n - t], r[t:]))
    return out


def main() -> int:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")

    from hemirl import policy as pol
    from hemirl.research_env import ResearchEnvConfig, build_research_env
    from hemirl.termination import TerminationConfig

    cfg = json.loads((paths.CONFIGS_ROOT / "train_healthy_v2.json").read_text(encoding="utf-8"))
    extra_cfg = HealthyRewardConfig(**cfg["extra_reward"])
    w_survival = float(cfg["reward_split"]["w_survival"])
    ckpt = paths.checkpoint_dir("LocomotionFull")

    env, raw = build_research_env(
        env_cfg=ResearchEnvConfig(
            termination=TerminationConfig(kind="research"),
            max_episode_seconds=20.0,
            keep_ledger=False,
            name="value_check",
        ),
        checkpoint_dir=ckpt,
    )
    stack = pol.load_sb3_stack(ckpt, env, deterministic=True)
    pid = int(env.pelvis_id)
    model, data = env.raw_env.model, env.raw_env.data

    obs, _ = env.reset(seed=0)
    R_off, R_new, comps, Z, A = [], [], [], [], []
    while True:
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
        survival = float(rc.get("survival_physical", 0.0))
        R_off.append(imitation + energy + healthy)
        R_new.append(imitation + energy + survival + extra_total)
        comps.append({"imitation": imitation, "energy": energy, "official_healthy": healthy,
                      "survival_physical": survival, **terms, "extra_total": extra_total,
                      "reward_returned": float(reward)})
        if terminated or truncated:
            break
    r_off = np.asarray(R_off, dtype=float)
    r_new = np.asarray(R_new, dtype=float)

    print(f"轨迹长度 = {len(r_off)} 步（{len(r_off) * 0.02:.2f} s）")
    print(f"official  奖励: mean={r_off.mean():+.3f}  sum={r_off.sum():+.1f}  "
          f"min={r_off.min():+.1f}  max={r_off.max():+.1f}")
    print(f"split     奖励: mean={r_new.mean():+.3f}  sum={r_new.sum():+.1f}  "
          f"min={r_new.min():+.1f}  max={r_new.max():+.1f}")
    print(f"前 5 步 official: {np.round(r_off[:5], 3).tolist()}")
    print(f"前 5 步 split   : {np.round(r_new[:5], 3).tolist()}")

    g_off_rec = mc_return_recursive(r_off)
    g_off_dir = mc_return_direct(r_off)
    g_new_rec = mc_return_recursive(r_new)
    g_new_dir = mc_return_direct(r_new)

    eps = 1e-9
    same_off = bool(np.max(np.abs(g_off_rec - g_off_dir)) < 1e-6)
    same_new = bool(np.max(np.abs(g_new_rec - g_new_dir)) < 1e-6)
    print(f"\n两种算法一致性: official={same_off}  split={same_new}")
    print(f"G_0(官方奖励) = {g_off_rec[0]:+.1f}")
    print(f"G_0(split奖励) = {g_new_rec[0]:+.1f}")
    print(f"若逐步奖励与该数组一致，G_0 的粗略量级 ≈ "
          f"{r_off.mean() * (1 - GAMMA ** len(r_off)) / (1 - GAMMA):+.1f}")

    # 回报一致性恒等式：Σ(1-γ)γ^t G_t ≈ Σ γ^t r_t
    def identity_check(r, g):
        pw = GAMMA ** np.arange(len(r))
        lhs = float(np.dot((1 - GAMMA) * pw, g))
        rhs = float(np.dot(pw, r))
        return lhs, rhs, abs(lhs - rhs)

    lhs_o, rhs_o, err_o = identity_check(r_off, g_off_rec)
    lhs_n, rhs_n, err_n = identity_check(r_new, g_new_rec)
    print(f"\n回报恒等式 official: LHS={lhs_o:+.3f}  RHS={rhs_o:+.3f}  误差={err_o:.2e}")
    print(f"回报恒等式 split   : LHS={lhs_n:+.3f}  RHS={rhs_n:+.3f}  误差={err_n:.2e}")

    # 与 env 实际返回的奖励对照（split 模式）
    r_env = np.asarray([c["reward_returned"] for c in comps], dtype=float)
    print(f"env 返回奖励 = split? {bool(np.allclose(r_env, r_new, atol=1e-6))}  "
          f"(mean={r_env.mean():+.3f})")

    verdict = {
        "n_steps": int(len(r_off)),
        "official_sum": float(r_off.sum()),
        "new_sum": float(r_new.sum()),
        "official_mean": float(r_off.mean()),
        "new_mean": float(r_new.mean()),
        "G0_official_recursive": float(g_off_rec[0]),
        "G0_official_direct": float(g_off_dir[0]),
        "G0_new_recursive": float(g_new_rec[0]),
        "G0_new_direct": float(g_new_dir[0]),
        "two_implementations_agree": bool(same_off and same_new),
        "identity_error_official": float(err_o),
        "identity_error_new": float(err_n),
        "env_reward_equals_split": bool(np.allclose(r_env, r_new, atol=1e-6)),
        "old_report_MC_official_t0": 4274.1,
        "old_report_MC_new_t0": 2859.1,
        "old_report_is_consistent": bool(
            abs(4274.1 - g_off_rec[0]) < 1.0 and abs(2859.1 - g_new_rec[0]) < 1.0
        ),
    }
    print("\n=== 结论 ===")
    print(f"  两种实现在本轨迹上一致: {verdict['two_implementations_agree']}")
    print(f"  旧报告的 MC 数值与本次重算是否吻合: {verdict['old_report_is_consistent']}")
    print(f"  重算 G_0: official={verdict['G0_official_recursive']:+.1f} "
          f"split={verdict['G0_new_recursive']:+.1f}")

    provenance.write_json(OUT, {
        "data": verdict,
        "provenance": provenance.run_provenance(
            entry="scripts/check_value_consistency.py", args={}, extra={}, checkpoint_dir=ckpt
        ),
    })
    env.close()
    print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
