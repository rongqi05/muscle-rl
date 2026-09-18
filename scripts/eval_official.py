"""行走策略评估入口（官方复现 / 研究环境）。

**两种模式的语义差别**（由 ``hemirl.research_env.ResearchLocomotionEnv`` 统一实现）：

=====================  ==============================  ====================================
模式                    终止                               截断
=====================  ==============================  ====================================
``--termination official``  ``not is_healthy``（偏离参考）       ``t >= terminate_time*cycles``
                        或 ``t >= T*cycles``                 （即 3.51 s）
``--termination research``  物理跌倒 / 明确任务失败            ``--max-episode-seconds``
                        （数值异常单独归类）                （默认 20 s）
=====================  ==============================  ====================================

示例::

    # 官方复现（保留原语义，用于与历史记录对比）
    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
        --episodes 5 --termination official --tag eval_short_baseline_official

    # 研究环境长时评估（20 s，不中途 reset、不覆盖根节点、无额外稳定力）
    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_official.py \
        --episodes 5 --termination research --max-episode-seconds 20 \
        --save-traj --save-ledger --video --tag eval_long_research

输出目录：``runs/<tag>/``，包含 ``run.json``（provenance）、``episodes.json``、
``summary.json``；``--save-traj`` 额外写轨迹 npz，``--save-ledger`` 写逐步 ledger。
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
from hemirl.muscle_actuator import StrengthSpec  # noqa: E402
from hemirl.research_env import ResearchEnvConfig  # noqa: E402
from hemirl.rollout import EvaluatorConfig, LocomotionEvaluator  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402


def build_termination(kind: str, cfg_path: Optional[Path]) -> TerminationConfig:
    if kind == "official":
        return TerminationConfig(kind="official")
    if cfg_path is not None and Path(cfg_path).is_file():
        return TerminationConfig.from_json(Path(cfg_path))
    return TerminationConfig(kind="research")


#: 汇总时求均值/标准差的指标
SUMMARY_KEYS = [
    "steps",
    "sim_time_s",
    "displacement_xy",
    "forward_displacement_x",
    "lateral_drift_y",
    "mean_speed_xy",
    "forward_speed_x",
    "mean_pelvis_height",
    "min_pelvis_height",
    "final_pelvis_height",
    "max_pelvis_up_tilt_deg",
    "final_pelvis_up_tilt_deg",
    "posture_change_deg",
    "max_root_speed",
    "max_root_ang_speed",
    "mean_qpos_track_err",
    "max_qpos_track_err",
    "final_qpos_track_err",
    "action_abs_mean",
    "activation_mean",
    "actuator_force_abs_mean",
    "ncon_mean",
    "total_normal_force_mean",
    "foot_contact_fraction",
    "max_abs_qvel",
    "jacobian_vs_index_max_diff",
]


def summarize(results: List[Dict[str, Any]], planned_seconds: Optional[float]) -> Dict[str, Any]:
    """对多个 episode 的关键指标做均值/标准差汇总，并**分开**统计成功与跌倒。"""
    out: Dict[str, Any] = {"n_episodes": len(results), "metrics": {}}
    for k in SUMMARY_KEYS:
        vals = [
            float(r[k])
            for r in results
            if r.get(k) is not None and np.isfinite(r.get(k, float("nan")))
        ]
        if not vals:
            continue
        out["metrics"][k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    fell = [r for r in results if r.get("termination_source") == "physical_fall"]
    numeric = [r for r in results if r.get("termination_source") == "numeric_anomaly"]
    completed = [
        r
        for r in results
        if r.get("termination_source") in ("time_limit", "official_time_limit")
    ]
    out["outcome_counts"] = {
        "physical_fall": len(fell),
        "numeric_anomaly": len(numeric),
        "completed_planned_duration": len(completed),
        "other": len(results) - len(fell) - len(numeric) - len(completed),
    }
    out["termination_reasons"] = _count_reasons(results)
    out["termination_sources"] = _count_sources(results)
    out["n_terminated"] = int(sum(1 for r in results if r.get("terminated")))
    out["n_truncated"] = int(sum(1 for r in results if r.get("truncated")))
    out["planned_duration_s"] = planned_seconds

    # 「成功行走」与「跌倒前运动」必须分开报告：跌倒前的瞬时速度不能当作效果改善
    if fell:
        out["fallen_only_speed_mean"] = float(
            np.mean([r["mean_speed_xy"] for r in fell if np.isfinite(r.get("mean_speed_xy", float("nan")))])
        )
        out["fallen_only_alive_time_mean"] = float(
            np.mean([r["sim_time_s"] for r in fell])
        )
        out["fallen_only_note"] = (
            "跌倒前平均速度只描述「跌倒前跑了多快」，不代表任务完成或效果改善"
        )
    if completed:
        out["completed_only_speed_mean"] = float(
            np.mean([r["mean_speed_xy"] for r in completed])
        )
        out["completed_only_alive_time_mean"] = float(np.mean([r["sim_time_s"] for r in completed]))
    return out


def _count_reasons(results: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in results:
        reason = r.get("termination_reason") or "未终止"
        out[reason] = out.get(reason, 0) + 1
    return out


def _count_sources(results: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in results:
        src = r.get("termination_source") or "none"
        out[src] = out.get(src, 0) + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="行走策略评估（官方复现 / 研究环境）")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="基础种子；第 i 个 episode 用 seed+i")
    parser.add_argument("--tag", type=str, default="eval_official")
    parser.add_argument("--out-root", type=str, default=str(paths.RUNS_ROOT))
    parser.add_argument("--termination", choices=["official", "research"], default="official")
    parser.add_argument(
        "--termination-config",
        type=str,
        default=str(paths.CONFIGS_ROOT / "termination_research.json"),
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=None,
        help="研究模式下的规定评估时长（秒）；None 表示沿用官方 3.51 s",
    )
    parser.add_argument("--max-control-steps", type=int, default=None, help="本地控制步上限（优先于时长）")
    parser.add_argument("--reward-mode", choices=["official", "split"], default="official")
    parser.add_argument("--paretic-side", choices=["L", "R"], default=None)
    parser.add_argument("--upper-scale", type=float, default=1.0)
    parser.add_argument("--lower-scale", type=float, default=1.0)
    parser.add_argument("--torso-scale", type=float, default=1.0)
    parser.add_argument("--include-shoulder-girdle", action="store_true")
    parser.add_argument(
        "--strength-mode",
        choices=["active_only", "active_and_passive"],
        default="active_only",
    )
    parser.add_argument("--stochastic", action="store_true", help="默认用确定性动作（tanh(mean)）")
    parser.add_argument("--save-traj", action="store_true")
    parser.add_argument("--save-ledger", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-max-frames", type=int, default=1100)
    parser.add_argument("--no-final-force-decomposition", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    term = build_termination(args.termination, Path(args.termination_config))
    research = ResearchEnvConfig(
        termination=term,
        max_episode_seconds=args.max_episode_seconds,
        max_control_steps=args.max_control_steps,
        reward_mode=args.reward_mode,
        name=f"eval_{args.termination}",
    )

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
        research=research,
        save_trajectory=args.save_traj,
        save_ledger=args.save_ledger,
        final_force_decomposition=not args.no_final_force_decomposition,
    )

    video_report: Dict[str, Any] = {"requested": bool(args.video)}
    results: List[Dict[str, Any]] = []
    planned_seconds = research.max_episode_seconds

    with LocomotionEvaluator(cfg) as ev:
        if planned_seconds is None:
            planned_seconds = ev.env.time_limit_s
        prov = provenance.run_provenance(
            entry="scripts/eval_official.py",
            args={k: v for k, v in vars(args).items()},
            extra={
                "evaluator_config": cfg.to_dict(),
                "research_env_config": research.to_dict(),
                "policy_effective_semantics": {
                    "deterministic": not args.stochastic,
                    "dynsyn_weight_amp": None,
                    "action_expansion": "DynSynLayer.repeat_replace_x（等价于按组索引展开）",
                },
                "muscle_map_counts": ev.mapping.counts(),
                "policy_load_notes": ev.stack.load_notes,
                "n_muscles": int(ev.model.nu),
                "nq": int(ev.model.nq),
                "nv": int(ev.model.nv),
                "control_dt": ev.dt,
                "frame_skip": ev.frame_skip,
                "physics_timestep": ev.env.physics_timestep,
                "official_time_limit_s": ev.env.official_time_limit_s,
                "effective_time_limit_s": ev.env.time_limit_s,
                "max_control_steps": ev.env.max_control_steps,
            },
            checkpoint_dir=Path(args.checkpoint_dir),
            loaded_model_path=ev.loaded_model_path,
            strength=spec.to_dict() if spec is not None else None,
            policy_inference={"deterministic": not args.stochastic, "device": cfg.device},
            seeds=[args.seed + i for i in range(args.episodes)],
        )
        provenance.write_json(out_dir / "run.json", prov)

        for i in range(args.episodes):
            seed = args.seed + i
            label = f"ep{i}"
            res = ev.run_episode(
                seed=seed,
                spec=spec,
                label=label,
                save_trajectory=args.save_traj,
                trail_dir=out_dir / "trajectories",
            )
            results.append(res.to_dict())
            print(
                f"[ep {i}] seed={seed} steps={res.steps} t={res.sim_time_s:.2f}s "
                f"src={res.termination_source} 终止={res.termination_reason} "
                f"前进x={res.forward_displacement_x:.3f}m 侧漂y={res.lateral_drift_y:.3f}m "
                f"v_xy={res.mean_speed_xy:.3f}m/s 最低骨盆={res.min_pelvis_height:.3f}m"
            )

        if args.video:
            from hemirl.render import record_video

            vcfg = EvaluatorConfig(
                checkpoint_dir=Path(args.checkpoint_dir),
                deterministic=not args.stochastic,
                termination=term,
                research=research,
                render_mode="rgb_array",
                final_force_decomposition=False,
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
            except Exception as exc:  # 渲染不可用不应中断实验
                video_report.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    summary = summarize(results, planned_seconds)
    summary["strength_spec"] = spec.to_dict() if spec is not None else None
    summary["termination_config"] = term.to_dict()
    summary["research_env_config"] = research.to_dict()
    summary["deterministic"] = not args.stochastic
    if args.video:
        summary["video"] = video_report

    provenance.write_json(out_dir / "episodes.json", {"episodes": results})
    provenance.write_json(out_dir / "summary.json", summary)

    print("\n=== 汇总 ===")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
    print("结局统计:", json.dumps(summary["outcome_counts"], ensure_ascii=False))
    print("终止来源:", json.dumps(summary["termination_sources"], ensure_ascii=False))
    if args.video:
        print("视频:", json.dumps(video_report, ensure_ascii=False))
    print(f"\n输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
