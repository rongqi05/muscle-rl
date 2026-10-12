"""失稳前 1–2 s 的**时间对齐**诊断（任务书第一阶段第 4 条）。

## 与 ``scripts/diagnose_instability.py`` 的区别

后者按**绝对时间**输出全部逐步信号（用于找「谁先越界」）；本脚本按
**「距离终止还有多少秒」**对齐，把多个回合叠在一起，回答的是一个更窄的问题：

> 在跌倒前的最后 2 s，**足部接触 / 地面反力 / 足部滑动 / 骨盆侧向位置与速度 /
> 左右下肢动作饱和 / 关节状态**这些量各自是什么样、谁在谁之前变化？

## 严格区分「观察到的时间顺序」与「已证实的因果」

输出里每一条领先关系都只标注为 **observed order**。要升级为因果，必须另有
单因素对照实验；本脚本**不**给因果结论。命名对照的既有结论（来自上一阶段
``reports/named_controls.json``）在本文件里只作为「已排除」引用，不重复。

运行::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/diagnose_prefix.py \
        --policies A=artifacts/checkpoints/LocomotionFull --seeds 301:305 \
        --window-s 2.0 --out reports/phase1_prefix_diag.json
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
from hemirl.termination import TerminationConfig  # noqa: E402

FOOT_BODIES = ("calcn_r", "calcn_l", "toes_r", "toes_l")
#: 下肢主要关节（用于看关节状态；名字取自 MS-Human-700 XML）
LOWER_JOINTS = (
    "hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
)


def parse_seeds(s: str) -> List[int]:
    if ":" in s:
        lo, hi = s.split(":")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split()]


def joint_index(model, name: str):
    """返回关节在 ``data.qpos`` 中的起始下标（找不到返回 None）。"""
    import mujoco

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        return None
    return int(model.jnt_qposadr[jid])


def run_policy(name: str, ckpt: Path, seeds: List[int], window_s: float,
               episode_s: float) -> Dict[str, Any]:
    import mujoco

    from hemirl import policy as pol
    from hemirl.rollout import root_state

    env, raw = build_research_env(
        env_cfg=ResearchEnvConfig(
            termination=TerminationConfig(kind="research"),
            max_episode_seconds=episode_s,
            keep_ledger=False,
            name=f"prefix_{name}",
        ),
        checkpoint_dir=ckpt,
    )
    model, data = raw.model, raw.data
    stack = pol.load_sb3_stack(ckpt, env, deterministic=True)
    pid = int(env.pelvis_id)
    foot_ids = {n: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)) for n in FOOT_BODIES}
    jidx = {j: joint_index(model, j) for j in LOWER_JOINTS}
    # 下肢肌肉在 actuator 维里的索引（用肌群映射，找 *_l / *_r 且属下肢）
    try:
        from hemirl import muscle_groups

        grp = muscle_groups.load_mapping()
        lower_muscles = set(grp.get("lower", []) or [])
    except Exception:  # noqa: BLE001
        lower_muscles = set()

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        rows: List[Dict[str, Any]] = []
        while True:
            z = np.asarray(stack.normalize_obs(obs), dtype=np.float32)
            action = np.asarray(stack.predict(z), dtype=float)
            obs, _rew, terminated, truncated, info = env.step(action)
            rs = root_state(model, data, pid)
            # 足部接触与地面反力（只计法向 > 20 N 的接触）
            buf = np.zeros(6, dtype=np.float64)
            gf: Dict[int, float] = {}
            for i in range(int(data.ncon)):
                con = data.contact[i]
                mujoco.mj_contactForce(model, data, i, buf)
                f = float(abs(buf[0]))
                for gid in (int(con.geom1), int(con.geom2)):
                    if f > gf.get(gid, 0.0):
                        gf[gid] = f
            loads, slips = {}, {}
            for nm, bid in foot_ids.items():
                if bid < 0:
                    continue
                adr, num = int(model.body_geomadr[bid]), int(model.body_geomnum[bid])
                fmax = max([gf.get(adr + g, 0.0) for g in range(num)] or [0.0])
                loads[nm] = fmax
                v = np.asarray(data.cvel[bid][3:6], dtype=float)
                slips[nm] = float(np.linalg.norm(v)) if fmax > 20.0 else 0.0
            sat = np.abs(action) > 0.99
            act = np.asarray(data.act, dtype=float).ravel()
            rows.append({
                "t": float(info["termination"]["elapsed_time_s"]),
                "pelvis_y": float(rs["pos"][1]),
                "ref_y": float(env._ref_y()),
                "pelvis_z": float(env.pelvis_height()),
                "lateral_vel": float(rs["lin_vel"][1]),
                "forward_vel": float(rs["lin_vel"][0]),
                "tilt_deg": float(env.up_tilt_deg()),
                "foot_load": loads,
                "foot_slip": slips,
                "n_loaded_feet": int(sum(1 for v in loads.values() if v > 20.0)),
                "sat_all": float(sat.mean()),
                "act_all": float(act.mean()),
                "joint_qpos": {j: (float(data.qpos[i]) if i is not None else None)
                               for j, i in jidx.items()},
            })
            if terminated or truncated:
                src = info["termination"]["termination_source"]
                reason = info["termination"]["termination_reason"]
                break
        per_seed.append({
            "seed": seed,
            "termination_source": src,
            "termination_reason": reason,
            "alive_time_s": rows[-1]["t"],
            "n_steps": len(rows),
            "rows": rows,
        })

    # ---------------------------------------------------------------- 对齐统计
    # 只对**因物理跌倒而结束**的回合做对齐叠加（时间截断/数值异常不参与）
    fell = [s for s in per_seed if s["termination_source"] == "physical_fall"]
    n_win = int(round(window_s / float(env.dt)))
    aligned: Dict[str, Dict[str, Any]] = {}
    if fell:
        keys = ["pelvis_y", "ref_y", "pelvis_z", "lateral_vel", "forward_vel", "tilt_deg",
                "sat_all", "act_all", "n_loaded_feet"]
        for k in keys:
            arr = []
            for s in fell:
                tail = s["rows"][-n_win:]
                v = [r[k] for r in tail]
                if len(v) < n_win:  # 前向填充（回合太短）
                    v = [v[0]] * (n_win - len(v)) + v
                arr.append(v)
            a = np.asarray(arr, dtype=float)
            aligned[k] = {
                "mean": [float(x) for x in a.mean(axis=0)],
                "p10": [float(x) for x in np.percentile(a, 10, axis=0)],
                "p90": [float(x) for x in np.percentile(a, 90, axis=0)],
            }
        for nm in FOOT_BODIES:
            arr = []
            for s in fell:
                tail = s["rows"][-n_win:]
                v = [r["foot_load"].get(nm, 0.0) for r in tail]
                if len(v) < n_win:
                    v = [v[0]] * (n_win - len(v)) + v
                arr.append(v)
            a = np.asarray(arr, dtype=float)
            aligned[f"load_{nm}"] = {"mean": [float(x) for x in a.mean(axis=0)]}
            arrs = []
            for s in fell:
                tail = s["rows"][-n_win:]
                v = [r["foot_slip"].get(nm, 0.0) for r in tail]
                if len(v) < n_win:
                    v = [v[0]] * (n_win - len(v)) + v
                arrs.append(v)
            a2 = np.asarray(arrs, dtype=float)
            aligned[f"slip_{nm}"] = {"mean": [float(x) for x in a2.mean(axis=0)]}

    # 相对跌倒时刻的领先量（观察到的顺序，不是因果）
    lead = {}
    for s in fell:
        rows = s["rows"]
        n = len(rows)
        base = rows[: max(1, n // 2)]
        w = min(n_win, n)
        tail = rows[-w:]
        for key, getter in (
            ("action_sat_all", lambda r: r["sat_all"]),
            ("lateral_vel_abs", lambda r: abs(r["lateral_vel"])),
            ("lateral_dev_abs", lambda r: abs(r["pelvis_y"] - r["ref_y"])),
            ("tilt_deg", lambda r: r["tilt_deg"]),
            ("no_foot_contact", lambda r: float(r["n_loaded_feet"] == 0)),
        ):
            vals = np.asarray([getter(r) for r in tail], dtype=float)
            b = np.asarray([getter(r) for r in base], dtype=float)
            mu, sd = float(b.mean()), float(b.std())
            thr = mu + 3.0 * sd if sd > 1e-12 else mu * 1.5 + 1e-9
            idx = np.where(vals > thr)[0]
            if len(idx):
                t_cross = float(tail[int(idx[0])]["t"])
                lead.setdefault(key, []).append(float(s["alive_time_s"] - t_cross))
    lead_summary = {
        k: {"n": len(v), "median_lead_s": float(np.median(v))}
        for k, v in lead.items()
    }

    env.close()
    return {
        "policy": name,
        "checkpoint_dir": str(ckpt),
        "n_seeds": len(seeds),
        "alive_times_s": [s["alive_time_s"] for s in per_seed],
        "termination_sources": {s["termination_source"]: sum(
            1 for x in per_seed if x["termination_source"] == s["termination_source"]
        ) for s in per_seed},
        "n_fell": len(fell),
        "window_s": window_s,
        "window_steps": n_win,
        "aligned": aligned,
        "lead_over_termination_s": lead_summary,
        "per_seed": per_seed,
    }


def main() -> int:
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")

    parser = argparse.ArgumentParser(description="失稳前时间对齐诊断")
    parser.add_argument("--policies", nargs="+", required=True, help="形如 A=<dir>")
    parser.add_argument("--seeds", default="301:305")
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--out", default=str(paths.REPORTS_ROOT / "phase1_prefix_diag.json"))
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    out: Dict[str, Any] = {"entry": "scripts/diagnose_prefix.py",
                           "seeds": seeds, "window_s": args.window_s, "policies": {}}
    for item in args.policies:
        name, _, d = item.partition("=")
        if not d:
            raise SystemExit(f"--policies 需要 name=dir，收到 {item!r}")
        res = run_policy(name, Path(d), seeds, args.window_s, args.episode_seconds)
        out["policies"][name] = res
        print(f"\n=== {name}: 存活 {np.mean(res['alive_times_s']):.2f}s "
              f"({res['termination_sources']}) ===")
        for k, v in sorted(res["lead_over_termination_s"].items(),
                           key=lambda kv: -kv[1]["median_lead_s"]):
            print(f"  {k:22s} 领先终止 {v['median_lead_s']:+.2f}s (n={v['n']})")
        if res["aligned"]:
            a = np.asarray(res["aligned"]["lateral_dev_abs"]["mean"]) if "lateral_dev_abs" in res["aligned"] \
                else None
            az = np.asarray(res["aligned"]["pelvis_z"]["mean"])
            av = np.asarray(res["aligned"]["lateral_vel"]["mean"])
            at = np.asarray(res["aligned"]["tilt_deg"]["mean"])
            w = len(az)
            print(f"  窗口对齐（{w} 步 = {w * 0.02:.2f}s，索引 0 = 终止前 {w * 0.02:.2f}s）：")
            for frac in (0.0, 0.25, 0.5, 0.75, 0.95):
                i = min(w - 1, int(frac * w))
                print(f"    t-{(w - 1 - i) * 0.02:4.2f}s  pelvis_z={az[i]:.3f}  "
                      f"lat_vel={av[i]:+.3f}  tilt={at[i]:5.1f}°"
                      + (f"  lat_dev={a[i]:.3f}" if a is not None else ""))

    out["provenance"] = provenance.run_provenance(
        entry="scripts/diagnose_prefix.py", args=vars(args), extra={},
    )
    provenance.write_json(Path(args.out), out)
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
