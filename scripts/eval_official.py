"""官方行走策略评估入口（最小案例 / 多重复）。

示例::

    # 最小案例：1 个 episode，官方终止规则
    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py --episodes 1 --tag eval_min

    # 5 个重复，研究终止规则，并保存轨迹
    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
        --episodes 5 --termination research --save-traj --tag eval_repeats

    # 患侧 R 上肢 0.5，下肢保持 1.0
    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
        --episodes 3 --paretic-side R --upper-scale 0.5 --lower-scale 1.0 \
        --termination research --tag eval_paretic_upper0.5

输出目录：``runs/<tag>/``，包含 ``run.json``（provenance）、``episodes.json``、
``summary.json``；加 ``--video`` 时输出视频或说明渲染失败原因。
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
from hemirl.muscle_actuator import StrengthSpec  # noqa: E402
from hemirl.rollout import EvaluatorConfig, LocomotionEvaluator  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402


def build_termination(kind: str, cfg_path: Path | None, time_limit: float | None) -> TerminationConfig:
    if kind == "official":
        return TerminationConfig(kind="official")
    if cfg_path is not None and Path(cfg_path).is_file():
        cfg = TerminationConfig.from_json(Path(cfg_path))
    else:
        cfg = TerminationConfig(kind="research")
    if time_limit is not None:
        object.__setattr__(cfg, "time_limit_s", time_limit)
    return cfg


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对多个 episode 的关键指标做均值/标准差汇总。"""
    keys = [
        "steps",
        "sim_time_s",
        "displacement_xy",
        "mean_speed_xy",
        "mean_pelvis_height",
        "min_pelvis_height",
        "max_pelvis_up_tilt_deg",
        "mean_qpos_track_err",
        "action_abs_mean",
        "activation_mean",
        "actuator_force_abs_mean",
        "ncon_mean",
        "total_normal_force_mean",
        "foot_contact_fraction",
        "max_abs_qvel",
    ]
    out: Dict[str, Any] = {"n_episodes": len(results), "metrics": {}}
    for k in keys:
        vals = [float(r[k]) for r in results if r.get(k) is not None and np.isfinite(r[k])]
        if not vals:
            continue
        out["metrics"][k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    term = {}
    for r in results:
        reason = r.get("termination_reason") or "未终止"
        term[reason] = term.get(reason, 0) + 1
    out["termination_reasons"] = term
    out["n_terminated"] = int(sum(1 for r in results if r.get("terminated")))
    out["n_truncated"] = int(sum(1 for r in results if r.get("truncated")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="官方行走策略评估")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="基础种子；第 i 个 episode 用 seed+i")
    parser.add_argument("--tag", type=str, default="eval_official")
    parser.add_argument("--out-root", type=str, default=str(paths.RUNS_ROOT))
    parser.add_argument("--termination", choices=["official", "research"], default="official")
    parser.add_argument("--termination-config", type=str, default=str(paths.CONFIGS_ROOT / "termination_research.json"))
    parser.add_argument("--time-limit", type=float, default=None, help="研究模式下覆盖时间上限（秒）")
    parser.add_argument("--paretic-side", choices=["L", "R"], default=None)
    parser.add_argument("--upper-scale", type=float, default=1.0)
    parser.add_argument("--lower-scale", type=float, default=1.0)
    parser.add_argument("--torso-scale", type=float, default=1.0)
    parser.add_argument("--include-shoulder-girdle", action="store_true")
    parser.add_argument(
        "--strength-mode",
        choices=["active_only", "active_and_passive"],
        default="active_only",
        help="主动力单独缩放，还是主动+被动力共同缩放",
    )
    parser.add_argument("--stochastic", action="store_true", help="默认用确定性动作（tanh(mean)）")
    parser.add_argument("--save-traj", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-max-frames", type=int, default=400)
    args = parser.parse_args()

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    term = build_termination(args.termination, Path(args.termination_config), args.time_limit)
    spec = None
    if args.paretic_side is not None:
        spec = StrengthSpec(
            paretic_side=args.paretic_side,
            upper_scale=args.upper_scale,
            lower_scale=args.lower_scale,
            torso_scale=args.torso_scale,
            include_shoulder_girdle=args.include_shoulder_girdle,
            mode=args.strength_mode,
        )

    cfg = EvaluatorConfig(
        checkpoint_dir=Path(args.checkpoint_dir),
        deterministic=not args.stochastic,
        termination=term,
        save_trajectory=args.save_traj,
    )

    video_report: Dict[str, Any] = {"requested": bool(args.video)}

    with LocomotionEvaluator(cfg) as ev:
        # provenance 先落盘，即使后面失败也有记录
        prov = provenance.build_provenance(
            extra={
                "entry": "scripts/eval_official.py",
                "args": vars(args),
                "evaluator_config": cfg.to_dict(),
                "muscle_map_counts": ev.mapping.counts(),
                "policy_load_notes": ev.stack.load_notes,
                "n_muscles": int(ev.model.nu),
                "nq": int(ev.model.nq),
                "nv": int(ev.model.nv),
                "control_dt": ev.dt,
                "frame_skip": ev.frame_skip,
            },
            checkpoint_dir=Path(args.checkpoint_dir),
        )
        provenance.write_json(out_dir / "run.json", prov)

        results = []
        for i in range(args.episodes):
            seed = args.seed + i
            label = f"ep{i}"
            res = ev.run_episode(
                seed=seed,
                spec=spec,
                label=label,
                termination=term,
                save_trajectory=args.save_traj,
                trail_dir=out_dir / "trajectories",
            )
            results.append(res.to_dict())
            print(
                f"[ep {i}] seed={seed} steps={res.steps} t={res.sim_time_s:.2f}s "
                f"终止={res.termination_reason} 位移={res.displacement_xy:.3f}m "
                f"平均速度={res.mean_speed_xy:.3f}m/s 最低骨盆={res.min_pelvis_height:.3f}m"
            )

        if args.video:
            from hemirl.render import record_video

            vcfg = EvaluatorConfig(
                checkpoint_dir=Path(args.checkpoint_dir),
                deterministic=not args.stochastic,
                termination=term,
                render_mode="rgb_array",
            )
            try:
                with LocomotionEvaluator(vcfg) as vev:
                    vr = record_video(
                        vev,
                        seed=args.seed,
                        out_path=out_dir / "rollout.mp4",
                        spec=spec,
                        max_frames=args.video_max_frames,
                    )
                video_report.update(vr.to_dict())
            except Exception as exc:
                video_report.update(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                )

    summary = summarize(results)
    summary["strength_spec"] = spec.to_dict() if spec is not None else None
    summary["termination_config"] = term.to_dict()
    summary["deterministic"] = not args.stochastic
    if args.video:
        summary["video"] = video_report

    provenance.write_json(out_dir / "episodes.json", {"episodes": results})
    provenance.write_json(out_dir / "summary.json", summary)

    print("\n=== 汇总 ===")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print("终止原因:", json.dumps(summary["termination_reasons"], ensure_ascii=False))
    if args.video:
        print("视频:", json.dumps(video_report, ensure_ascii=False))
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
