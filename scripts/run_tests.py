"""统一测试入口（验收的一部分）。

用法::

    PYTHONPATH=. python scripts/run_tests.py               # 核心 + 研究环境回归
    PYTHONPATH=. python scripts/run_tests.py --with-heavy   # 另外运行肌力验证脚本

**任一步失败即返回非零退出码**，便于作为验收命令的一环。

注意：策略一致性验证（``scripts/verify_policy.py``）需要 checkpoint，因此不在这里运行，
而是由 ``scripts/acceptance.py`` 在检测到 checkpoint 后调用。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-heavy", action="store_true", help="同时运行 scripts/verify_strength.py")
    parser.add_argument("--skip-env", action="store_true", help="跳过需要真实环境的回归测试")
    args = parser.parse_args()

    from tests import test_core

    rc_core = test_core.main()
    if rc_core != 0:
        print("\n核心单元测试失败")
        return rc_core

    if not args.skip_env:
        from tests import test_research_env

        print()
        rc_env = test_research_env.main()
        if rc_env != 0:
            print("\n研究环境回归测试失败")
            return rc_env

    if args.with_heavy:
        print("\n=== 运行 scripts/verify_strength.py ===")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_strength.py")],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        if r.returncode != 0:
            print("verify_strength 失败")
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
