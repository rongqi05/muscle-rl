"""长时行走失稳诊断入口。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/diagnose_instability.py \
        --seeds 0 1 2 3 4 --max-episode-seconds 20

输出 ``reports/instability_diag.json``：

* 每个种子的结局、观测分组的裁剪统计、**领先指标排序**（谁先越界）；
* 其中一个代表种子的完整逐步序列（用于画图/复核）。

报告里的关键区分（不要混淆）：

* ``ref_*``：**参考自己**的位置波动；反映上游轨迹的固有特性；
* ``drift_*``：**人体相对参考**的偏差；反映人体是否真的在积累误差。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402
from hemirl.instability_diag import DiagConfig, InstabilityDiagnoser  # noqa: E402
from hemirl.research_env import ResearchEnvConfig  # noqa: E402
from hemirl.rollout import EvaluatorConfig, LocomotionEvaluator  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402

#: 打印与汇总时优先关注的信号
KEY_SIGNALS = (
    "pelvis_y",
    "pelvis_vy",
    "up_tilt_deg",
    "pelvis_w_norm",
    "com_vy",
    "foot_slip_max",
    "no_support",
    "act_asym_lower",
    "action_sat_frac",
    "action_sat_frac_L_lower",
    "action_sat_frac_R_lower",
    "drift_y_human_minus_ref",
    "ref_y_minus_ref_y0",
    "qpos_track_err",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="长时行走失稳诊断")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-episode-seconds", type=float, default=20.0)
    parser.add_argument("--baseline-window", type=float, default=1.0)
    parser.add_argument("--k-sigma", type=float, default=6.0)
    parser.add_argument("--series-seed", type=int, default=0, help="保存哪个种子的完整逐步序列")
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "instability_diag.json"))
    args = parser.parse_args()

    term = TerminationConfig(kind="research")
    cfg = EvaluatorConfig(
        checkpoint_dir=Path(args.checkpoint_dir),
        deterministic=True,
        termination=term,
        research=ResearchEnvConfig(
            termination=term,
            max_episode_seconds=args.max_episode_seconds,
            name="instability_diag",
        ),
        final_force_decomposition=False,
    )

    report: dict = {
        "entry": "scripts/diagnose_instability.py",
        "args": vars(args),
        "episodes": [],
        "series": {},
    }

    with LocomotionEvaluator(cfg) as ev:
        diag = InstabilityDiagnoser(
            ev,
            DiagConfig(
                dt=ev.dt,
                baseline_window_s=args.baseline_window,
                k_sigma=args.k_sigma,
                clip_obs=float(getattr(ev.stack.vec_normalize, "clip_obs", 10.0)),
            ),
        )
        vn = ev.stack.vec_normalize
        report["vec_normalize"] = {
            "clip_obs": float(getattr(vn, "clip_obs", float("nan"))),
            "norm_obs": bool(getattr(vn, "norm_obs", False)),
            "norm_reward": bool(getattr(vn, "norm_reward", False)),
            "count": float(getattr(vn.obs_rms, "count", float("nan"))),
            "load_mode": ev.stack.load_notes.get("vec_normalize_load_mode"),
        }

        for seed in args.seeds:
            r = diag.run(seed)
            if seed == args.series_seed:
                report["series"] = r.pop("series")
            else:
                r.pop("series", None)
            report["episodes"].append(r)
            print(
                f"[seed {seed}] 存活={r['alive_time_s']:.2f}s steps={r['n_steps']} "
                f"src={r['termination_source']} {r['termination_reason']}"
            )

        report["provenance"] = provenance.run_provenance(
            entry="scripts/diagnose_instability.py",
            args=vars(args),
            checkpoint_dir=Path(args.checkpoint_dir),
            loaded_model_path=ev.loaded_model_path,
        )

    provenance.write_json(Path(args.out), report)

    print("\n=== 观测分组裁剪统计（各 seed 最大值；基于**裁剪前** z_raw）===")
    names = list(report["episodes"][0]["obs_group_summary"].keys())
    for n in names:
        cf = max(e["obs_group_summary"][n]["clip_fraction_max"] for e in report["episodes"])
        nc = max(e["obs_group_summary"][n]["n_clipped_max"] for e in report["episodes"])
        fc = [e["obs_group_summary"][n]["first_clip_t"] for e in report["episodes"]]
        fc = [x for x in fc if x is not None]
        za = max(e["obs_group_summary"][n]["z_raw_absmax_max"] for e in report["episodes"])
        print(
            f"  {n:20s} n_clip_max={nc:6.0f}  frac_max={cf:7.4f}  "
            f"z_raw_absmax={za:9.2f}  首次裁剪={min(fc) if fc else None}"
        )

    print("\n=== 领先指标（首次越界时间，各 seed）===")
    import numpy as _np

    rows: Dict[str, List[float]] = {}
    for e in report["episodes"]:
        for t0, name in e["leading_indicators"]:
            if name in KEY_SIGNALS:
                rows.setdefault(name, []).append(float(t0))
    for name, ts in sorted(rows.items(), key=lambda kv: float(_np.median(kv[1]))):
        print(
            f"  {name:30s} 中位首次越界 = {float(_np.median(ts)):5.2f}s  "
            f"(各 seed: {[round(x, 2) for x in sorted(ts)]})"
        )

    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
