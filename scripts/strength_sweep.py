"""肌力扫描实验：固定策略权重，扫描上肢 / 下肢患侧肌力倍率。

* **扫描 A**：下肢倍率固定 1.0，上肢倍率 ∈ {1.0, 0.75, 0.5, 0.25}
* **扫描 B**：上肢倍率固定 1.0，下肢倍率 ∈ {1.0, 0.75, 0.5, 0.25}

同一 seed 下的所有倍率使用**相同的初始条件**（配对种子）：环境 reset 的随机性完全由
`env.reset(seed=...)` 决定，且策略与场景不变，因此差异只来自肌力缩放。

重要说明：这些结果只反映「固定权重策略对肌力变化的响应」，
**不能**解释为已经适应损伤的患者策略。

示例::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/strength_sweep.py \
        --paretic-side R --seeds 0 1 2 --termination research --tag sweep_R
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402
from hemirl.muscle_actuator import StrengthSpec  # noqa: E402
from hemirl.rollout import EvaluatorConfig, LocomotionEvaluator  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402

DEFAULT_LEVELS = (1.0, 0.75, 0.5, 0.25)


def build_levels() -> List[Dict[str, Any]]:
    """返回 [(sweep, upper_scale, lower_scale)] 列表。"""
    out: List[Dict[str, Any]] = []
    for u in DEFAULT_LEVELS:
        out.append({"sweep": "A_upper", "upper_scale": u, "lower_scale": 1.0})
    for lo in DEFAULT_LEVELS:
        out.append({"sweep": "B_lower", "upper_scale": 1.0, "lower_scale": lo})
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> Path:
    keys = [
        "sweep",
        "label",
        "seed",
        "upper_scale",
        "lower_scale",
        "n_scaled_muscles",
        "steps",
        "sim_time_s",
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
        "terminated",
        "truncated",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="患侧肌力倍率扫描")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--paretic-side", choices=["L", "R"], default="R")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tag", type=str, default="sweep")
    parser.add_argument("--out-root", type=str, default=str(paths.RUNS_ROOT))
    parser.add_argument("--termination", choices=["official", "research"], default="research")
    parser.add_argument(
        "--strength-mode",
        choices=["active_only", "active_and_passive"],
        default="active_only",
    )
    parser.add_argument("--include-shoulder-girdle", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save-traj", action="store_true")
    parser.add_argument("--no-baseline", action="store_true", help="跳过每个扫描的 1.0 基准（1.0 已在扫描内）")
    args = parser.parse_args()

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    term = TerminationConfig(kind=args.termination)
    cfg = EvaluatorConfig(
        checkpoint_dir=Path(args.checkpoint_dir),
        deterministic=not args.stochastic,
        termination=term,
        save_trajectory=args.save_traj,
    )
    levels = build_levels()

    all_rows: List[Dict[str, Any]] = []
    with LocomotionEvaluator(cfg) as ev:
        prov = provenance.build_provenance(
            extra={
                "entry": "scripts/strength_sweep.py",
                "args": vars(args),
                "levels": levels,
                "paretic_side": args.paretic_side,
                "strength_mode": args.strength_mode,
                "muscle_map_counts": ev.mapping.counts(),
                "policy_load_notes": ev.stack.load_notes,
                "termination_config": term.to_dict(),
            },
            checkpoint_dir=Path(args.checkpoint_dir),
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
                    seed=seed,
                    spec=spec,
                    label=label,
                    termination=term,
                    save_trajectory=args.save_traj,
                    trail_dir=out_dir / "trajectories",
                )
                row = res.to_dict()
                row["sweep"] = sweep
                row["upper_scale"] = u
                row["lower_scale"] = lo
                row["n_scaled_muscles"] = int(row["strength"]["n_scaled"])
                row["group_counts"] = row["strength"].get("group_counts")
                row["applied_slots"] = row["strength"].get("applied_slots")
                all_rows.append(row)
                print(
                    f"[{sweep}] u={u:<5} l={lo:<5} seed={seed} "
                    f"steps={res.steps:3d} t={res.sim_time_s:.2f}s "
                    f"disp={res.displacement_xy:.3f}m v={res.mean_speed_xy:.3f}m/s "
                    f"minz={res.min_pelvis_height:.3f}m term={res.termination_reason}"
                )

    provenance.write_json(out_dir / "episodes.json", {"episodes": all_rows})
    write_csv(out_dir / "sweep.csv", all_rows)

    # 每个 (sweep, level) 的聚合
    agg: Dict[str, Any] = {}
    for sweep in sorted({r["sweep"] for r in all_rows}):
        agg[sweep] = {}
        for level_val in sorted({r["upper_scale"] if sweep == "A_upper" else r["lower_scale"] for r in all_rows if r["sweep"] == sweep}, reverse=True):
            sel = [
                r
                for r in all_rows
                if r["sweep"] == sweep
                and (r["upper_scale"] if sweep == "A_upper" else r["lower_scale"]) == level_val
            ]
            def m(key):
                vals = [float(r[key]) for r in sel if r.get(key) is not None and np.isfinite(r[key])]
                return float(np.mean(vals)) if vals else None
            agg[sweep][f"{level_val:g}"] = {
                "n": len(sel),
                "scale": level_val,
                "steps_mean": m("steps"),
                "sim_time_mean": m("sim_time_s"),
                "displacement_mean": m("displacement_xy"),
                "speed_mean": m("mean_speed_xy"),
                "min_pelvis_height_mean": m("min_pelvis_height"),
                "qpos_track_err_mean": m("mean_qpos_track_err"),
                "activation_mean": m("activation_mean"),
                "force_abs_mean": m("actuator_force_abs_mean"),
                "n_terminated": int(sum(1 for r in sel if r["terminated"])),
                "reasons": sorted({r["termination_reason"] for r in sel if r["termination_reason"]}),
            }
    provenance.write_json(out_dir / "summary.json", {"aggregate": agg, "n_episodes": len(all_rows)})

    print("\n=== 聚合结果 ===")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"\n输出: {out_dir}  (sweep.csv / episodes.json / summary.json / run.json)")


if __name__ == "__main__":
    main()
