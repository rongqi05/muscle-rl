"""肌力扫描实验：固定策略权重，扫描上肢 / 下肢患侧肌力倍率。

* **扫描 A**：下肢倍率固定 1.0，上肢倍率 ∈ {1.0, 0.75, 0.5, 0.25}
* **扫描 B**：上肢倍率固定 1.0，下肢倍率 ∈ {1.0, 0.75, 0.5, 0.25}

同一 seed 下的所有倍率使用**相同的初始条件**（配对种子）：reset 的随机性完全由
``env.reset(seed=...)`` 决定，策略与场景不变，因此差异只来自肌力缩放。

## 两条必须遵守的口径

1. **区分「成功走完」与「跌倒前运动」**。跌倒前那一段的平均速度只说明「跌之前跑得多快」，
   不能当作效果改善。汇总里两类分开给数（``completed_*`` / ``fallen_*``）。
2. **终止语义用研究环境**（``hemirl.research_env``）：``terminated`` = 物理跌倒，
   ``truncated`` = 达到规定时长。默认时长与原短时协议一致（官方 ``terminate_time*cycles``
   = 3.51 s），可用 ``--max-episode-seconds`` 做长时对照。

重要说明：这些结果只反映「固定权重策略对肌力变化的响应」，
**不能**解释为已经适应损伤的患者策略。

示例::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/strength_sweep.py \
        --paretic-side R --seeds 0 1 2 3 4 --termination research --tag sweep_R_research
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402
from hemirl.muscle_actuator import StrengthSpec  # noqa: E402
from hemirl.research_env import ResearchEnvConfig  # noqa: E402
from hemirl.rollout import EvaluatorConfig, LocomotionEvaluator  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402

DEFAULT_LEVELS = (1.0, 0.75, 0.5, 0.25)

CSV_KEYS = [
    "sweep",
    "label",
    "seed",
    "upper_scale",
    "lower_scale",
    "n_scaled_muscles",
    "outcome",
    "steps",
    "sim_time_s",
    "forward_displacement_x",
    "lateral_drift_y",
    "displacement_xy",
    "mean_speed_xy",
    "mean_pelvis_height",
    "min_pelvis_height",
    "max_pelvis_up_tilt_deg",
    "mean_qpos_track_err",
    "activation_mean",
    "actuator_force_abs_mean",
    "action_abs_mean",
    "foot_contact_fraction",
    "ncon_mean",
    "total_normal_force_mean",
    "termination_reason",
    "termination_source",
    "terminated",
    "truncated",
]


def build_levels() -> List[Dict[str, Any]]:
    """返回 [(sweep, upper_scale, lower_scale)] 列表。"""
    out: List[Dict[str, Any]] = []
    for u in DEFAULT_LEVELS:
        out.append({"sweep": "A_upper", "upper_scale": u, "lower_scale": 1.0})
    for lo in DEFAULT_LEVELS:
        out.append({"sweep": "B_lower", "upper_scale": 1.0, "lower_scale": lo})
    return out


def outcome_of(row: Dict[str, Any]) -> str:
    """把一次 episode 的结局归成三类，避免把「跌倒前运动」混进成功样本。"""
    src = row.get("termination_source")
    if src in ("time_limit", "official_time_limit"):
        return "completed"
    if src == "physical_fall":
        return "fell"
    if src == "numeric_anomaly":
        return "numeric_anomaly"
    return "unfinished"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_KEYS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _mean(vals: List[float]) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def aggregate(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按 (sweep, level) 聚合，并**分开**给成功/跌倒两类统计。"""
    agg: Dict[str, Any] = {}
    for sweep in sorted({r["sweep"] for r in all_rows}):
        agg[sweep] = {}
        levels = sorted(
            {r["upper_scale"] if sweep == "A_upper" else r["lower_scale"]
             for r in all_rows if r["sweep"] == sweep},
            reverse=True,
        )
        for level_val in levels:
            sel = [
                r
                for r in all_rows
                if r["sweep"] == sweep
                and (r["upper_scale"] if sweep == "A_upper" else r["lower_scale"]) == level_val
            ]
            completed = [r for r in sel if r["outcome"] == "completed"]
            fell = [r for r in sel if r["outcome"] == "fell"]
            entry: Dict[str, Any] = {
                "n": len(sel),
                "scale": level_val,
                "n_completed": len(completed),
                "n_fell": len(fell),
                "n_numeric_anomaly": sum(1 for r in sel if r["outcome"] == "numeric_anomaly"),
                "alive_time_mean_s": _mean([r["sim_time_s"] for r in sel]),
                "alive_time_mean_s_completed_only": _mean([r["sim_time_s"] for r in completed]),
                "alive_time_mean_s_fallen_only": _mean([r["sim_time_s"] for r in fell]),
                "steps_mean": _mean([r["steps"] for r in sel]),
                "forward_displacement_mean_m": _mean([r["forward_displacement_x"] for r in sel]),
                "lateral_drift_mean_m": _mean([r["lateral_drift_y"] for r in sel]),
                # 全样本速度：**只用于「平均运动强度」描述，不用于效果比较**
                "speed_mean_all_mps": _mean([r["mean_speed_xy"] for r in sel]),
                "speed_mean_completed_only_mps": _mean([r["mean_speed_xy"] for r in completed]),
                "speed_mean_fallen_only_mps": _mean([r["mean_speed_xy"] for r in fell]),
                "min_pelvis_height_mean_m": _mean([r["min_pelvis_height"] for r in sel]),
                "qpos_track_err_mean": _mean([r["mean_qpos_track_err"] for r in sel]),
                "activation_mean": _mean([r["activation_mean"] for r in sel]),
                "force_abs_mean": _mean([r["actuator_force_abs_mean"] for r in sel]),
                "reasons": sorted({r["termination_reason"] for r in sel if r["termination_reason"]}),
                "interpretation": (
                    "speed_mean_fallen_only_mps 只描述「跌倒前跑得多快」，"
                    "不能解释为效果改善；只有 speed_mean_completed_only_mps 与 "
                    "n_completed 才有「走完」的含义"
                ),
            }
            agg[sweep][f"{level_val:g}"] = entry
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="患侧肌力倍率扫描")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--paretic-side", choices=["L", "R"], default="R")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tag", type=str, default="sweep")
    parser.add_argument("--out-root", type=str, default=str(paths.RUNS_ROOT))
    parser.add_argument("--termination", choices=["official", "research"], default="research")
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=None,
        help="研究模式的时间上限；None 表示与原短时协议一致（官方 3.51 s）",
    )
    parser.add_argument(
        "--strength-mode",
        choices=["active_only", "active_and_passive"],
        default="active_only",
    )
    parser.add_argument("--include-shoulder-girdle", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save-traj", action="store_true")
    parser.add_argument("--save-ledger", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    term = TerminationConfig(kind=args.termination)
    research = ResearchEnvConfig(
        termination=term,
        max_episode_seconds=args.max_episode_seconds,
        name=f"sweep_{args.termination}",
    )
    cfg = EvaluatorConfig(
        checkpoint_dir=Path(args.checkpoint_dir),
        deterministic=not args.stochastic,
        termination=term,
        research=research,
        save_trajectory=args.save_traj,
        save_ledger=args.save_ledger,
    )
    levels = build_levels()

    all_rows: List[Dict[str, Any]] = []
    with LocomotionEvaluator(cfg) as ev:
        prov = provenance.run_provenance(
            entry="scripts/strength_sweep.py",
            args=vars(args),
            extra={
                "levels": levels,
                "paretic_side": args.paretic_side,
                "strength_mode": args.strength_mode,
                "research_env_config": research.to_dict(),
                "muscle_map_counts": ev.mapping.counts(),
                "policy_load_notes": ev.stack.load_notes,
                "policy_inference": {"deterministic": not args.stochastic},
                "effective_time_limit_s": ev.env.time_limit_s,
                "max_control_steps": ev.env.max_control_steps,
                "official_time_limit_s": ev.env.official_time_limit_s,
            },
            checkpoint_dir=Path(args.checkpoint_dir),
            loaded_model_path=ev.loaded_model_path,
            seeds=list(args.seeds),
        )
        provenance.write_json(out_dir / "run.json", prov)

        for level in levels:
            sweep, u, lo = level["sweep"], level["upper_scale"], level["lower_scale"]
            for seed in args.seeds:
                spec = StrengthSpec(
                    paretic_side=args.paretic_side,
                    upper_scale=u,
                    lower_scale=lo,
                    include_shoulder_girdle=args.include_shoulder_girdle,
                    mode=args.strength_mode,
                )
                label = f"{sweep}_u{u}_l{lo}"
                res = ev.run_episode(
                    seed=seed, spec=spec, label=label, save_trajectory=args.save_traj,
                    trail_dir=out_dir / "trajectories",
                )
                row = res.to_dict()
                row["sweep"] = sweep
                row["upper_scale"] = u
                row["lower_scale"] = lo
                row["n_scaled_muscles"] = int(row["strength"]["n_scaled"])
                row["group_counts"] = row["strength"].get("group_counts")
                row["applied_slots"] = row["strength"].get("applied_slots")
                row["outcome"] = outcome_of(row)
                all_rows.append(row)
                print(
                    f"[{sweep}] u={u:<5} l={lo:<5} seed={seed} outcome={row['outcome']:<10} "
                    f"steps={res.steps:3d} t={res.sim_time_s:.2f}s "
                    f"x={res.forward_displacement_x:.3f}m v_xy={res.mean_speed_xy:.3f}m/s "
                    f"minz={res.min_pelvis_height:.3f}m term={res.termination_reason}"
                )

    provenance.write_json(out_dir / "episodes.json", {"episodes": all_rows})
    write_csv(out_dir / "sweep.csv", all_rows)
    agg = aggregate(all_rows)
    provenance.write_json(
        out_dir / "summary.json",
        {
            "aggregate": agg,
            "n_episodes": len(all_rows),
            "paretic_side": args.paretic_side,
            "strength_mode": args.strength_mode,
            "max_episode_seconds": args.max_episode_seconds,
            "outcome_totals": {
                k: sum(1 for r in all_rows if r["outcome"] == k)
                for k in ("completed", "fell", "numeric_anomaly", "unfinished")
            },
        },
    )

    print("\n=== 聚合结果（成功 / 跌倒分开）===")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"\n输出: {out_dir}  (sweep.csv / episodes.json / summary.json / run.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
