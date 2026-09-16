"""导出肌肉分组映射清单（CSV + JSON 摘要）。

用法::

    PYTHONPATH=. python scripts/export_muscle_map.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import muscle_groups, paths  # noqa: E402


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser(description="导出 MS-Human-700 肌肉分组映射")
    parser.add_argument("--xml", type=str, default=str(paths.MODEL_XML))
    parser.add_argument("--out-dir", type=str, default=str(paths.REPORTS_ROOT))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    m = muscle_groups.build_map(model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = m.write_csv(out_dir / "muscle_group_map.csv")
    summary = m.to_json()
    summary["xml"] = args.xml
    summary["problems"] = muscle_groups.validate_group_consistency(m)
    (out_dir / "muscle_group_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[saved] {csv_path}")
    print(f"[saved] {out_dir / 'muscle_group_summary.json'}")

    # 抽查：左右侧对称性与上下肢举例
    def show(entries) -> None:
        for e in entries:
            print(
                f"  {e.index:3d} {e.name:16s} group={e.group_key:10s} "
                f"src={e.source_file} crossed={e.crossed_limbs} bodies={e.bodies}"
            )

    print("\n抽查（下肢 R / L 各 3 个）:")
    show(m.by_group("R", "lower")[:3] + m.by_group("L", "lower")[:3])
    print("抽查（上肢 R / L 各 3 个）:")
    show(m.by_group("R", "upper")[:3] + m.by_group("L", "upper")[:3])
    print("抽查（躯干前 3 个）:")
    show(m.by_group("M", "torso")[:3])
    print("\n抽查（肩胛带：权威归 torso、但只跨越上肢关节，可用 include_shoulder_girdle 归入上肢）:")
    show([e for e in m.entries if e.limb_group == "torso" and e.crossed_limbs == ["upper"]][:3])


if __name__ == "__main__":
    main()
