"""参考轨迹连续性检查：循环、前进位移累积、未来参考观测。

长时评估**之前**必须做这一步。上游 ``LocomotionCycleTrajectory.query_batch`` 用

    qpos[:, 0] += cycle_number * qpos_traj[-1, 0]
    qpos[:, 2] += cycle_number * qpos_traj[-1, 2]

做跨周期位移累积，其中 ``cycle_number = floor(t / terminate_time)``。如果
``terminate_time`` 不是恰好一个周期，或者 ``qpos_traj[-1]`` 不是恰好一个步幅，
位移就会随周期数**线性漂移**；那种情况下长时评估测到的是「参考本身在漂」，
而不是策略的平衡能力。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/probe_reference_continuity.py --cycles 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402
from hemirl.envs import OfficialEnvConfig, build_env  # noqa: E402
from hemirl.research_env import reference_continuity_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="参考轨迹连续性检查")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "reference_continuity.json"))
    args = parser.parse_args()

    env_cfg = OfficialEnvConfig.from_checkpoint(Path(args.checkpoint_dir))
    env, raw = build_env(env_cfg)
    try:
        env.reset(seed=0)
        rep = reference_continuity_report(raw.unwrapped, n_cycles=args.cycles)
    finally:
        env.close()

    rep["code_version"] = provenance.code_version()
    rep["model_xml"] = {
        "path": str(paths.MODEL_XML),
        "sha256": provenance.file_sha256(paths.MODEL_XML),
    }
    provenance.write_json(Path(args.out), rep)

    print("=== 参考轨迹连续性（按控制步长采样）===")
    for k in (
        "terminate_time_s",
        "framerate_hz",
        "num_frames",
        "frame_span_s",
        "control_step_s",
        "wrap_frame_gap_s",
        "max_step_change_within_cycle",
        "max_step_change_at_cycle_boundary",
        "boundary_over_within_ratio",
        "boundary_is_comparable_to_one_step",
        "pose_wrap_discontinuity_max_rad",
        "cycle_is_strictly_periodic",
        "per_cycle_forward_from_qpos_m",
        "per_cycle_forward_from_xpos_m",
        "stride_m",
        "forward_accumulation_rel_err",
        "future_ref_step_change_max",
        "cumulative_pose_wrap_over_eval_rad",
        "cumulative_forward_gap_over_eval_m",
        "needs_loop_guard",
    ):
        print(f"  {k:40s} = {rep.get(k)}")
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
