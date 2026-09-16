"""动力学闭环审计入口。

示例::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/audit_dynamics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import dynamics_audit, paths, provenance  # noqa: E402
from hemirl.rollout import EvaluatorConfig, LocomotionEvaluator  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="动力学闭环审计")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "dynamics_audit.json"))
    args = parser.parse_args()

    report: dict = {}

    print("=== A. 静态代码审计 ===")
    static = dynamics_audit.static_code_audit()
    report["static_code_audit"] = static
    for k in (
        "kinematic_play_default",
        "step_lines",
        "kinematic_play_lines",
        "step_state_assignments",
        "step_reference_buffer_assignments",
        "reset_model_calls_set_state",
        "reset_model_line",
        "pd_or_extra_torque_patterns",
    ):
        print(f"  {k}: {json.dumps(static.get(k), ensure_ascii=False)}")
    print(f"  gymnasium _step_mujoco_simulation (line {static['gymnasium_step_mujoco_simulation_line']}): {static['gymnasium_step_mujoco_simulation_body']}")

    cfg = EvaluatorConfig(
        checkpoint_dir=Path(args.checkpoint_dir),
        deterministic=True,
        termination=TerminationConfig(kind="research"),
    )

    print("\n=== B. 模型结构审计 ===")
    with LocomotionEvaluator(cfg) as ev:
        struct = dynamics_audit.model_structure_audit(ev.model)
        report["model_structure_audit"] = struct
        print("  counts:", json.dumps(struct["counts"]))
        print("  actuators:", json.dumps(struct["actuators"], ensure_ascii=False))
        print("  root:", json.dumps(struct["root"], ensure_ascii=False))
        print("  joint_types:", json.dumps(struct["joint_types"]))
        print("  equality types:", json.dumps(struct["equality"]["types"]))
        print("  equality coupled_to:", json.dumps(struct["equality"]["coupled_to"], ensure_ascii=False))
        print("  passive:", json.dumps(struct["passive"], ensure_ascii=False))
        print("  n contact geoms:", len(struct["contact_geoms"]))
        print("  opt:", json.dumps(struct["opt"], ensure_ascii=False))

        print("\n=== C. 运行时审计（策略驱动） ===")
        rt = dynamics_audit.runtime_audit(ev, n_steps=args.steps, zero_action=False)
        for k, v in rt.items():
            if k != "series":
                print(f"  {k}: {v}")
        report["runtime_policy_driven"] = rt

        print("\n=== C2. 运行时审计（零肌肉激励，验证重力主导） ===")
        rt0 = dynamics_audit.runtime_audit(ev, n_steps=args.steps, zero_action=True)
        for k, v in rt0.items():
            if k != "series":
                print(f"  {k}: {v}")
        report["runtime_zero_action"] = rt0

    report["provenance"] = provenance.build_provenance(
        extra={"entry": "scripts/audit_dynamics.py", "args": vars(args)},
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    out = provenance.write_json(Path(args.out), report)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
