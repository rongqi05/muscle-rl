"""统一测试入口。

用法::

    PYTHONPATH=. python scripts/run_tests.py            # 单元测试
    PYTHONPATH=. python scripts/run_tests.py --with-heavy   # 另外运行肌力验证脚本
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-heavy", action="store_true", help="同时运行 scripts/verify_strength.py")
    args = parser.parse_args()

    from tests import test_core

    rc = test_core.main()
    if rc != 0:
        print("\n单元测试失败，跳过后续")
        return rc

    if args.with_heavy:
        print("\n=== 运行 scripts/verify_strength.py ===")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_strength.py")],
            cwd=str(ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        )
        if r.returncode != 0:
            print("verify_strength 失败")
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
