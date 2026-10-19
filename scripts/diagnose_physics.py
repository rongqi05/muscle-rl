"""阶段 B：物理失稳诊断（失稳前 2 s 的时间对齐）。

在固定 v1 checkpoint、正常肌力、确定性推理、冻结 VecNormalize 下，对开发种子做
逐控制步记录：

* 仿真时间与参考相位；
* 左右足接触状态、左右足竖直地面反力、总接触法向力；
* 全身质心与骨盆的高度、竖直速度；侧向位置与速度；
* 髋（屈/内收/旋转）、膝、踝关节的**肌肉驱动关节力矩**（``data.qfrc_actuator``）；
* 关键支撑肌群（臀大/臀中/股四/腘绳/比目鱼/腓肠/胫前）的激活与力；
* 奖励分项与终止原因。

输出：

1. ``reports/phase6_physics_diag.json`` —— 对齐统计（末 2 s 的 mean/p10/p90）、
   每个物理量相对「前 1/3 基线」的首次越界时刻（**时间顺序，非因果**）；
2. ``reports/phase6_physics_diag.png`` —— 时间对齐多面板图；
3. 逐种子表（控制台与 JSON）。

**只用于定位原因**：本脚本不做任何干预；任何外加力/扰动实验若做，另行命名且
不得进入最终策略训练/验收。

运行::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/diagnose_physics.py \
        --policy runs/train/healthy_v1 --seeds 301:305 --window-s 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402

FOOT_BODIES = ("calcn_r", "calcn_l", "toes_r", "toes_l")
#: 主关节力矩（qvel dof）
TORQUE_JOINTS = (
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
)
#: 关键支撑肌群（激活与力；多组分的肌肉求和）
MUSCLE_GROUPS = {
    "glmax": ("glmax1", "glmax2", "glmax3"),
    "glmed": ("glmed1", "glmed2", "glmed3"),
    "vas": ("vaslat", "vasmed", "vasint"),
    "hams": ("bflh", "bfsh", "semimem", "semiten"),
    "soleus": ("soleus",),
    "gastroc": ("gaslat", "gasmed"),
    "tibant": ("tibant",),
}


def parse_seeds(s: str) -> List[int]:
    if ":" in s:
        lo, hi = s.split(":")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split()]


def find_muscle_ids(model, prefix: str) -> List[int]:
    import mujoco

    ids = []
    for i in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if n and n.startswith(prefix):
            ids.append(i)
    return ids


def run_policy(ckpt: Path, seeds: List[int], window_s: float,
               episode_s: float = 20.0, lead_window_s: float = 4.0) -> Dict[str, Any]:
    import mujoco

    from hemirl import policy as pol
    from hemirl.rollout import root_state
    from hemirl.research_env import ResearchEnvConfig, build_research_env
    from hemirl.termination import TerminationConfig

    env, raw = build_research_env(
        env_cfg=ResearchEnvConfig(termination=TerminationConfig(kind="research"),
                                  max_episode_seconds=episode_s, keep_ledger=False,
                                  name="physics_diag"),
        checkpoint_dir=ckpt,
    )
    model, data = raw.model, raw.data
    stack = pol.load_sb3_stack(ckpt, env, deterministic=True)
    pid = int(env.pelvis_id)
    period = float(raw.trajectory.get_trajectory_properties(0)[0])
    foot_ids = {n: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)) for n in FOOT_BODIES}
    tq = {}
    for jname in TORQUE_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        tq[jname] = int(model.jnt_dofadr[jid]) if jid >= 0 else None
    msk = {}
    for gname, prefixes in MUSCLE_GROUPS.items():
        for side in ("_r", "_l"):
            ids = [i for p in prefixes for i in find_muscle_ids(model, p + side)]
            msk[f"{gname}{side}"] = ids

    rows_all: List[Dict[str, Any]] = []
    prev_com_z: Optional[float] = None
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        prev_com_z = float(np.asarray(data.subtree_com[0])[2])
        rows: List[Dict[str, Any]] = []
        while True:
            z = np.asarray(stack.normalize_obs(obs), dtype=np.float32)
            action = np.asarray(stack.predict(z), dtype=float)
            obs, _rew, terminated, truncated, info = env.step(action)
            rs = root_state(model, data, pid)
            com = np.asarray(data.subtree_com[0])
            com_vz = float((com[2] - prev_com_z) / env.dt)
            prev_com_z = float(com[2])

            # 足部竖直 GRF（世界 z = 法向力 × 接触 frame 的法向 z 分量）
            buf = np.zeros(6, dtype=np.float64)
            load: Dict[str, Dict[str, float]] = {}
            total_normal = 0.0
            for i in range(int(data.ncon)):
                con = data.contact[i]
                mujoco.mj_contactForce(model, data, i, buf)
                fn = float(abs(buf[0]))
                total_normal += fn
                fz = fn * float(con.frame[2])
                for gid in (int(con.geom1), int(con.geom2)):
                    for nm, bid in foot_ids.items():
                        if bid < 0:
                            continue
                        adr, num = int(model.body_geomadr[bid]), int(model.body_geomnum[bid])
                        if adr <= gid < adr + num:
                            cur = load.setdefault(nm, {"fn": 0.0, "fz": 0.0})
                            cur["fn"] = max(cur["fn"], fn)
                            cur["fz"] = max(cur["fz"], fz)
            row: Dict[str, Any] = {
                "t": float(info["termination"]["elapsed_time_s"]),
                "phase": float((float(info["termination"]["elapsed_time_s"]) % period) / period),
                "ref_y": float(env._ref_y()),
                "pelvis_z": float(rs["pos"][2]),
                "pelvis_vz": float(rs["lin_vel"][2]),
                "com_z": float(com[2]),
                "com_vz": com_vz,
                "pelvis_y": float(rs["pos"][1]),
                "pelvis_vy": float(rs["lin_vel"][1]),
                "load_r": load.get("calcn_r", {}).get("fn", 0.0) + load.get("toes_r", {}).get("fn", 0.0),
                "load_l": load.get("calcn_l", {}).get("fn", 0.0) + load.get("toes_l", {}).get("fn", 0.0),
                "fz_r": load.get("calcn_r", {}).get("fz", 0.0) + load.get("toes_r", {}).get("fz", 0.0),
                "fz_l": load.get("calcn_l", {}).get("fz", 0.0) + load.get("toes_l", {}).get("fz", 0.0),
                "total_contact_normal": total_normal,
                "torque": {j: (float(data.qfrc_actuator[d]) if d is not None else None)
                           for j, d in tq.items()},
            }
            act_all = np.asarray(data.act, dtype=float).ravel()
            force_all = np.asarray(data.actuator_force, dtype=float).ravel()
            for gname, ids in msk.items():
                if ids:
                    row[f"act_{gname}"] = float(act_all[ids].mean())
                    row[f"force_{gname}"] = float(force_all[ids].mean())
                else:
                    row[f"act_{gname}"] = None
                    row[f"force_{gname}"] = None
            row["reward"] = {k: float(v) for k, v in (info.get("reward_components") or {}).items()
                             if isinstance(v, (int, float)) and k not in ("reward_split_config",)}
            rows.append(row)
            if terminated or truncated:
                rows[-1]["termination_source"] = info["termination"]["termination_source"]
                rows[-1]["termination_reason"] = info["termination"]["termination_reason"]
                break
        rows_all.append({"seed": seed, "rows": rows,
                         "termination_source": rows[-1].get("termination_source")})

    # ---------------------------------------------------------------- 对齐统计
    fell = [s for s in rows_all if s["termination_source"] == "physical_fall"]
    n_win = int(round(window_s / float(env.dt)))
    # 越界检测用更宽的窗口（2 s 可能截断真实的首越界）
    n_lead = int(round(lead_window_s / float(env.dt)))
    scalar_keys = ["pelvis_z", "pelvis_vz", "com_z", "com_vz", "pelvis_y", "pelvis_vy",
                   "load_r", "load_l", "total_contact_normal"]
    aligned: Dict[str, Dict[str, Any]] = {}
    for k in scalar_keys:
        aligned[k] = align(fell, k, n_win)
    for k in ("fz_r", "fz_l"):
        aligned[k] = align(fell, k, n_win)
    for k in list(tq.keys()):
        pass
    for j in TORQUE_JOINTS:
        arr = []
        for s in fell:
            v = [r["torque"].get(j) or 0.0 for r in s["rows"][-n_win:]]
            arr.append(pad(v, n_win))
        a = np.asarray(arr, dtype=float)
        aligned[f"torque_{j}"] = {"mean": list(a.mean(axis=0))}
    for gname in msk.keys():
        for prefix in ("act", "force"):
            arr = []
            for s in fell:
                v = [r.get(f"{prefix}_{gname}") or 0.0 for r in s["rows"][-n_win:]]
                arr.append(pad(v, n_win))
            a = np.asarray(arr, dtype=float)
            aligned[f"{prefix}_{gname}"] = {"mean": list(a.mean(axis=0))}

    # ---------------------------------------------------------- 首次越界时刻
    # 单侧量（向下/向外偏离才算）：高度下降、侧向速度绝对值等
    cross_keys = {
        "lateral_dev": lambda r: abs(r["pelvis_y"] - r["ref_y"]),
        "lateral_vel": lambda r: abs(r["pelvis_vy"]),
        "com_vz_down": lambda r: max(0.0, -r["com_vz"]),
        "pelvis_vz_down": lambda r: max(0.0, -r["pelvis_vz"]),
        "com_height": lambda r: -r["com_z"],
        "pelvis_height": lambda r: -r["pelvis_z"],
        "total_contact": lambda r: -r["total_contact_normal"],
        "load_imbalance": lambda r: abs(r["load_r"] - r["load_l"]),
    }
    # 双侧量（增或减都算异常）：激活/肌力/力矩的绝对偏离
    two_sided = {}
    for gname in msk.keys():
        two_sided[f"act_{gname}"] = (lambda r, g=gname: r.get(f"act_{g}") or 0.0)
        two_sided[f"force_{gname}"] = (lambda r, g=gname: r.get(f"force_{g}") or 0.0)
    for j in TORQUE_JOINTS:
        two_sided[f"torque_{j}"] = (lambda r, j=j: r["torque"].get(j) or 0.0)

    leads: Dict[str, List[float]] = {}
    for s in fell:
        rows = s["rows"]
        n = len(rows)
        base = rows[: max(1, n // 3)]
        tail = rows[-n_lead:]
        alive = s["rows"][-1]["t"]
        for k, getter in {**cross_keys, **two_sided}.items():
            vals = [v for v in (getter(r) for r in tail) if v is not None]
            bvals = [v for v in (getter(r) for r in base) if v is not None]
            if len(vals) < n_lead or not bvals:
                continue
            b = np.asarray(bvals, dtype=float)
            mu, sd = float(b.mean()), float(b.std())
            thr = 3.0 * sd if sd > 1e-12 else 1e-9
            v = np.asarray(vals, dtype=float)
            dev = np.abs(v - mu) if k in two_sided else (v - mu)
            idx = np.where(dev > thr)[0]
            if len(idx):
                t_cross = float(tail[int(idx[0])]["t"])
                leads.setdefault(k, []).append(float(alive - t_cross))
    lead_summary = {
        k: {"n": len(v), "median_lead_s": float(np.median(v)),
            "min_lead_s": float(np.min(v))}
        for k, v in sorted(leads.items(), key=lambda kv: -np.median(kv[1]))
    }
    env.close()
    return {
        "policy": str(ckpt),
        "period_s": period,
        "n_seeds": len(seeds),
        "n_fell": len(fell),
        "window_s": window_s,
        "window_steps": n_win,
        "aligned": aligned,
        "lead_over_termination_s": lead_summary,
        "per_seed": [{k: v for k, v in s.items() if k != "rows"} | {"rows": s["rows"]}
                     for s in rows_all],
    }


def pad(v: List[float], n: int) -> List[float]:
    if len(v) >= n:
        return list(v[-n:])
    return [v[0]] * (n - len(v)) + list(v)


def align(fell: List[Dict[str, Any]], key: str, n_win: int) -> Dict[str, Any]:
    arr = []
    for s in fell:
        v = [r.get(key) or 0.0 for r in s["rows"][-n_win:]]
        arr.append(pad(v, n_win))
    a = np.asarray(arr, dtype=float)
    return {
        "mean": [float(x) for x in a.mean(axis=0)],
        "p10": [float(x) for x in np.percentile(a, 10, axis=0)],
        "p90": [float(x) for x in np.percentile(a, 90, axis=0)],
    }


def make_figure(res: Dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = res["window_steps"]
    t = -(np.arange(n)[::-1] + 1) * 0.02
    al = res["aligned"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax = axes[0]
    ax.plot(t, al["load_r"]["mean"], label="GRF_r")
    ax.plot(t, al["load_l"]["mean"], label="GRF_l")
    ax.plot(t, al["total_contact_normal"]["mean"], label="total")
    ax.set_ylabel("GRF (N)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(t, al["pelvis_z"]["mean"], label="pelvis_z")
    ax.plot(t, al["com_z"]["mean"], label="com_z")
    ax.set_ylabel("height (m)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[2]
    ax.plot(t, al["pelvis_vz"]["mean"], label="pelvis_vz")
    ax.plot(t, al["com_vz"]["mean"], label="com_vz")
    ax.plot(t, al["pelvis_vy"]["mean"], label="pelvis_vy(lat)")
    ax.set_ylabel("vel (m/s)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[3]
    for j in TORQUE_JOINTS:
        ax.plot(t, al[f"torque_{j}"]["mean"], label=j, lw=1)
    ax.set_ylabel("qfrc_actuator (Nm)")
    ax.legend(ncol=2, fontsize=6); ax.grid(alpha=0.3)
    ax.set_xlabel("距终止 (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def main() -> int:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=str(paths.RUNS_ROOT / "train" / "healthy_v1"))
    parser.add_argument("--seeds", default="301:305")
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--lead-window-s", type=float, default=4.0)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--out-prefix", default="phase6_physics_diag",
                        help="输出文件名前缀（reports/<prefix>.json / .png）")
    parser.add_argument("--keep-rows-seeds", type=int, default=2,
                        help="JSON 里保留逐逐步行的种子数（其余只留统计）")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    res = run_policy(Path(args.policy), seeds, args.window_s, args.episode_seconds,
                     args.lead_window_s)
    print(f"=== {res['policy']}：{res['n_fell']}/{res['n_seeds']} 回合跌倒 ===")
    print("领先量（相对终止，中位）:")
    for k, v in res["lead_over_termination_s"].items():
        print(f"  {k:24s} {v['median_lead_s']:+6.2f}s (n={v['n']})")
    print("\n逐种子：")
    for s in res["per_seed"]:
        last = s["rows"][-1]
        print(f"  seed={s['seed']} alive={last['t']:.2f}s src={s['termination_source']}")

    out_json = paths.REPORTS_ROOT / f"{args.out_prefix}.json"
    out_png = paths.REPORTS_ROOT / f"{args.out_prefix}.png"
    make_figure(res, out_png)
    res["figure"] = str(out_png)
    # 控制体积：逐种子逐步行只保留前 N 个种子
    for i, s in enumerate(res["per_seed"]):
        if i >= args.keep_rows_seeds:
            s["rows_dropped"] = len(s["rows"])
            s["rows"] = []
    res["provenance"] = provenance.run_provenance(
        entry="scripts/diagnose_physics.py", args=vars(args), extra={},
        checkpoint_dir=Path(args.policy),
    )
    provenance.write_json(out_json, res)
    print(f"\n[saved] {out_json}\n[saved] {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
