"""检查 tendon 结构：用于建立数据驱动的肌肉-部位映射。

用法::

    PYTHONPATH=. python scripts/inspect_tendons.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402

# MuJoCo 的 tendon 类型：0 = fixed, 1 = spatial（部分 Python 绑定未导出 mjtTendon 枚举）
TENDON_TYPE_NAMES = {0: "mjTENDON_FIXED", 1: "mjTENDON_SPATIAL"}


def _tendon_type_name(v: int) -> str:
    return TENDON_TYPE_NAMES.get(int(v), f"unknown({v})")


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=str(paths.MODEL_XML))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    rep: dict = {"xml": args.xml, "ntendon": int(model.ntendon)}

    ttypes = Counter()
    for t in range(model.ntendon):
        ttypes[_tendon_type_name(int(model.tendon_type[t]))] += 1
    rep["tendon_type_counts"] = dict(ttypes)

    # 一个空间 tendon 的 wrap 组成
    sample = []
    for t in range(min(3, model.ntendon)):
        wrapadr = int(model.tendon_wrapadr[t])
        wrapnum = int(model.tendon_wrapnum[t])
        wraps = []
        for w in range(wrapadr, wrapadr + wrapnum):
            objid = int(model.wrap_objid[w])
            objtype = int(model.wrap_objtype[w])
            name = mujoco.mj_id2name(model, mujoco.mjtObj(objtype), objid)
            wraps.append({"objtype": objtype, "objid": objid, "name": name})
        sample.append(
            {
                "tendon": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, t),
                "type": _tendon_type_name(int(model.tendon_type[t])),
                "wrapnum": wrapnum,
                "wraps": wraps,
                "adr": int(model.tendon_adr[t]),
            }
        )
    rep["tendon_samples"] = sample

    # actuator -> tendon 映射覆盖情况
    trnid = np.asarray(model.actuator_trnid)
    trntype = np.asarray(model.actuator_trntype)
    rep["actuator_trntype_counts"] = dict(Counter(int(v) for v in trntype))
    rep["trntype_names"] = {
        str(int(v)): mujoco.mjtTrn(int(v)).name for v in sorted(set(int(x) for x in trntype))
    }

    # wrap 对象类型统计（0=body,1=geom? 以 mjtObj 为准）
    wrap_objtype = Counter()
    for t in range(model.ntendon):
        a, b = int(model.tendon_wrapadr[t]), int(model.tendon_wrapnum[t])
        for w in range(a, a + b):
            wrap_objtype[mujoco.mjtObj(int(model.wrap_objtype[w])).name] += 1
    rep["wrap_objtype_counts"] = dict(wrap_objtype)

    # 每个 actuator 的 tendon 覆盖到的 body 集合规模
    sizes = []
    for a in range(model.nu):
        if int(trntype[a]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            sizes.append(-1)
            continue
        t = int(trnid[a, 0])
        a0, b0 = int(model.tendon_wrapadr[t]), int(model.tendon_wrapnum[t])
        bodies = set()
        for w in range(a0, a0 + b0):
            if mujoco.mjtObj(int(model.wrap_objtype[w])) == mujoco.mjtObj.mjOBJ_BODY:
                bodies.add(int(model.wrap_objid[w]))
            elif mujoco.mjtObj(int(model.wrap_objtype[w])) == mujoco.mjtObj.mjOBJ_SITE:
                bodies.add(int(model.site_bodyid[int(model.wrap_objid[w])]))
            elif mujoco.mjtObj(int(model.wrap_objtype[w])) == mujoco.mjtObj.mjOBJ_GEOM:
                bodies.add(int(model.geom_bodyid[int(model.wrap_objid[w])]))
        sizes.append(len(bodies))
    rep["tendon_body_coverage"] = {
        "n_actuators": len(sizes),
        "n_no_tendon": int(sum(1 for s in sizes if s == -1)),
        "min": int(min(s for s in sizes if s >= 0)),
        "max": int(max(sizes)),
        "mean": float(np.mean([s for s in sizes if s >= 0])),
    }

    # 展示若干上肢/下肢肌肉的 tendon→body 名单
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    for want in ["addbrev_r", "bflh_r", "soleus_r", "tibant_r", "deltoid_r", "biceps_r", "trap_r"]:
        if want not in names:
            continue
        a = names.index(want)
        t = int(trnid[a, 0])
        a0, b0 = int(model.tendon_wrapadr[t]), int(model.tendon_wrapnum[t])
        bodies = []
        for w in range(a0, a0 + b0):
            pt, oi = int(model.wrap_objtype[w]), int(model.wrap_objid[w])
            if mujoco.mjtObj(pt) == mujoco.mjtObj.mjOBJ_BODY:
                bid = oi
            elif mujoco.mjtObj(pt) == mujoco.mjtObj.mjOBJ_SITE:
                bid = int(model.site_bodyid[oi])
            else:
                bid = int(model.geom_bodyid[oi])
            bodies.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid))
        rep.setdefault("example_muscles", {})[want] = sorted(set(bodies))

    text = json.dumps(rep, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
