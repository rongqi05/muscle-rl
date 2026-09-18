"""策略一致性验证入口（官方 SB3 路径 vs 纯 torch 直读路径）。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/verify_policy.py

判定：``max |a_sb3 - a_direct| <= --tol``（默认 1e-5，仅容许 float32 噪声）。
**失败时返回非零退出码**，并打印最早出现差异的计算环节。

报告写入 ``reports/verify_policy.json``，字段包含 ``passed`` / ``tol`` /
``max_abs_action_diff`` / ``n_samples`` / ``code_version`` / ``checkpoint_sha256``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402
from hemirl.policy_verify import DEFAULT_TOL, PolicyVerifyConfig, verify_policy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="策略加载一致性验证")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "verify_policy.json"))
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--n-reset", type=int, default=6, help="reset 观测样本数")
    parser.add_argument("--n-traj", type=int, default=120, help="行走轨迹观测样本数")
    parser.add_argument("--traj-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-amp-experiment", action="store_true", help="跳过非官方 DynSyn 命名实验")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = PolicyVerifyConfig(
        checkpoint_dir=Path(args.checkpoint_dir),
        tol=args.tol,
        n_reset=args.n_reset,
        n_traj=args.n_traj,
        traj_seeds=tuple(args.traj_seeds),
        device=args.device,
        run_amp_experiment=not args.no_amp_experiment,
    )
    report = verify_policy(cfg, out_path=Path(args.out), verbose=not args.quiet)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
