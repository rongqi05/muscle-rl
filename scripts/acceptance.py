"""验收入口：一条命令跑完本阶段的全部验证，并**准确传播失败退出码**。

用法::

    PYTHONPATH=. MUJOCO_GL=egl python scripts/acceptance.py

包含（按顺序，任一步失败立即以非零退出码结束）：

1. 核心单元测试（``tests/test_core.py``，18 项）
2. 研究环境回归（``tests/test_research_env.py``，11 项：终止步/截断/跌倒/官方参考终止/数值异常/
   终止后不推进/计时来源/奖励拆分/根速度 Jacobian/约束力分解/参考连续性）
3. 肌力参数化验证（``scripts/verify_strength.py``）
4. 策略一致性验证（``scripts/verify_policy.py``）—— 需要 checkpoint；缺失时记为
   ``skipped`` 并默认返回非零（可用 ``--allow-missing-checkpoint`` 放宽）

结果写入 ``reports/acceptance.json``，逐项标注 ``passed`` / ``failed`` / ``skipped``。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hemirl import paths  # noqa: E402


def _checkpoint_present(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "checkpoint" / "best_model.zip").is_file() and (
        checkpoint_dir / "checkpoint" / "best_env.zip"
    ).is_file()


def run_step(name: str, cmd: List[str], why: str = "") -> Dict[str, Any]:
    print(f"\n{'=' * 78}\n[{name}] {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    dt = time.perf_counter() - t0
    status = "passed" if proc.returncode == 0 else "failed"
    print(f"[{name}] -> {status}（returncode={proc.returncode}, {dt:.1f}s）", flush=True)
    return {
        "name": name,
        "cmd": cmd,
        "why": why,
        "status": status,
        "returncode": int(proc.returncode),
        "wall_time_s": float(dt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="第一阶段验收")
    parser.add_argument("--checkpoint-dir", type=str, default=str(paths.checkpoint_dir("LocomotionFull")))
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "acceptance.json"))
    parser.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        help="checkpoint 缺失时只记 skipped 而不让验收失败",
    )
    parser.add_argument("--skip-heavy", action="store_true", help="跳过肌力验证（约 1 分钟）")
    args = parser.parse_args()

    py = sys.executable
    ckpt = Path(args.checkpoint_dir)
    steps: List[Dict[str, Any]] = []

    # 1) 核心单元测试
    steps.append(
        run_step(
            "unit_tests",
            [py, str(ROOT / "scripts" / "run_tests.py")],
            "肌群映射 / 肌力缩放语义 / F0 槽位 / 终止配置 / 组展开 / numpy 兼容层 + 研究环境回归",
        )
    )
    if steps[-1]["status"] == "failed":
        return _finish(steps, Path(args.out))

    # 2) 肌力验证
    if not args.skip_heavy:
        steps.append(
            run_step(
                "verify_strength",
                [py, str(ROOT / "scripts" / "verify_strength.py")],
                "F0 槽位实证、无累乘、组外不变、按关节名构造姿态的 F_active 缩放、被动通道对照",
            )
        )
        if steps[-1]["status"] == "failed":
            return _finish(steps, Path(args.out))

    # 3) 策略一致性（需要 checkpoint）
    if _checkpoint_present(ckpt):
        steps.append(
            run_step(
                "verify_policy",
                [py, str(ROOT / "scripts" / "verify_policy.py"), "--checkpoint-dir", str(ckpt)],
                "官方 SB3 路径 vs 纯 torch 直读路径逐元素一致（含行走轨迹观测）",
            )
        )
    else:
        print(f"\n[verify_policy] SKIPPED：缺少 checkpoint（{ckpt / 'checkpoint'}）")
        steps.append(
            {
                "name": "verify_policy",
                "cmd": None,
                "why": "官方 SB3 路径 vs 纯 torch 直读路径一致性",
                "status": "skipped",
                "returncode": None,
                "skip_reason": f"缺少 checkpoint 文件: {ckpt / 'checkpoint' / 'best_model.zip'}",
            }
        )

    return _finish(steps, Path(args.out), allow_missing=args.allow_missing_checkpoint)


def _finish(steps: List[Dict[str, Any]], out: Path, allow_missing: bool = False) -> int:
    from hemirl import provenance

    n_failed = sum(1 for s in steps if s["status"] == "failed")
    n_skipped = sum(1 for s in steps if s["status"] == "skipped")
    n_passed = sum(1 for s in steps if s["status"] == "passed")
    failed_required = n_failed > 0
    skipped_required = n_skipped > 0 and not allow_missing
    payload = {
        "generated_at_utc": provenance.build_provenance()["generated_at_utc"],
        "code_version": provenance.code_version(),
        "steps": steps,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "ok": not failed_required and not skipped_required,
        "note": (
            "skipped 也需要显式处理：默认把「该跑但没跑」视为验收未完成，"
            "可用 --allow-missing-checkpoint 放宽"
        ),
    }
    provenance.write_json(out, payload)
    print(f"\n{'=' * 78}")
    for s in steps:
        mark = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[s["status"]]
        print(f"  [{mark}] {s['name']}")
    print(f"  通过 {n_passed} / 失败 {n_failed} / 跳过 {n_skipped}  ->  {out}")
    if failed_required:
        print("  验收失败")
    elif skipped_required:
        print("  验收未完成（存在被跳过的必需项）")
    else:
        print("  验收通过")
    return 1 if (failed_required or skipped_required) else 0


if __name__ == "__main__":
    sys.exit(main())
