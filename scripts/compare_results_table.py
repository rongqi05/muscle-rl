"""把多份 ``eval_report.json`` 汇总成交付用的对比表（策略 × 时长）。

任务书要求：「表格比较官方、v1、每个新策略在相同条件下的 6 s、10 s、20 s 表现；
逐项列出终止来源、20 s 成功数、速度、侧偏、倾角、滑动和异常数量」。

本脚本只做**汇总与格式化**，不做任何重新计算，避免与评估脚本产生第二套口径。

运行::

    PYTHONPATH=. python scripts/compare_results_table.py \
        --inputs A=... B=... --out reports/phase5_table.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402

COLUMNS = [
    ("mean_alive_time_s", "存活(s)", "{:.2f}"),
    ("n_completed", "走满", "{}/{}"),
    ("complete_rate", "走满率", "{:.2f}"),
    ("n_fell", "跌倒", "{}"),
    ("n_numeric_anomaly", "异常", "{}"),
    ("mean_forward_speed", "速度(m/s)", "{:.3f}"),
    ("mean_target_speed_error", "速度误差", "{:.3f}"),
    ("mean_max_lateral_dev_m", "侧偏(m)", "{:.3f}"),
    ("mean_max_pelvis_tilt_deg", "倾角(°)", "{:.1f}"),
    ("mean_foot_slip_mps", "滑步(m/s)", "{:.2f}"),
    ("mean_activation", "激活", "{:.4f}"),
    ("mean_action_saturation", "饱和", "{:.4f}"),
    ("n_degenerate_stood_still", "退化", "{}"),
]


def load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(agg: Dict[str, Any], key: str, template: str) -> str:
    v = agg.get(key)
    if v is None:
        return "-"
    if template == "{}/{}":
        return f"{int(agg.get('n_completed', 0))}/{int(agg.get('n_episodes', 0))}"
    try:
        return template.format(v)
    except (ValueError, TypeError):
        return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总评测对比表")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="形如 名称=路径/eval_report.json；名称里用 @ 分隔策略与时长，如 A@6s=...")
    parser.add_argument("--out", default=str(paths.REPORTS_ROOT / "phase5_table.md"))
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for item in args.inputs:
        label, _, p = item.partition("=")
        if not p:
            raise SystemExit(f"--inputs 需要 名称=路径，收到 {item!r}")
        d = load(Path(p))
        pol, _, secs = label.partition("@")
        for name, pol_data in d["policies"].items():
            rows.append({
                "group": pol,
                "seconds": secs or f"{d['protocol']['episode_seconds']:.0f}s",
                "policy": name,
                "agg": pol_data["aggregate"],
                "gate": pol_data["aggregate"].get("gate", {}),
                "sources": pol_data["aggregate"].get("completion_by_source", {}),
            })

    # 去重（同一个 group/seconds/policy 只留一条）
    seen = set()
    uniq = []
    for r in rows:
        k = (r["group"], r["seconds"], r["policy"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    uniq.sort(key=lambda r: (r["group"], r["seconds"], r["policy"]))

    header = ["组", "时长", "策略"] + [c[1] for c in COLUMNS] + ["终止来源", "门槛"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in uniq:
        cells = [r["group"], r["seconds"], r["policy"]]
        cells += [fmt(r["agg"], k, t) for k, _t, t in [(c[0], c[1], c[2]) for c in COLUMNS]]
        cells.append(", ".join(f"{k}×{v}" for k, v in sorted(r["sources"].items())) or "-")
        g = r["gate"]
        cells.append(f"{g.get('n_ok', '-')}/{g.get('n_required', '-')}")
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    md = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    print(md)

    provenance.write_json(out.with_suffix(".json"), {
        "generated_from": [i for i in args.inputs],
        "n_rows": len(uniq),
        "rows": [{k: v for k, v in r.items() if k != "agg"} | {"aggregate": r["agg"]} for r in uniq],
    })
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
