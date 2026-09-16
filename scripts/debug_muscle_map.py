"""调试：检查指定肌肉的 tendon wrap → body → 跨越关节链路。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hemirl import muscle_groups, paths  # noqa: E402


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(paths.MODEL_XML))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    trnid = m.actuator_trnid

    for want in ["MF_m1s_r", "IO1_r", "MF_m1_laminar_r", "addbrev_r", "LD_L1_r", "soleus_r"]:
        if want not in names:
            print(want, "NOT FOUND")
            continue
        a = names.index(want)
        t = int(trnid[a, 0])
        adr, num = int(m.tendon_adr[t]), int(m.tendon_num[t])
        print(f"\n=== {want} (act {a}) tendon={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, t)} adr={adr} num={num}")
        bodies = []
        for w in range(adr, adr + num):
            wt = int(m.wrap_type[w])
            oi = int(m.wrap_objid[w])
            if wt == 3:
                b = int(m.site_bodyid[oi])
                sname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, oi)
                print(f"  wrap {w}: SITE {sname} -> body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)} (id {b})")
                bodies.append(b)
            elif wt in (4, 5):
                b = int(m.geom_bodyid[oi])
                gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, oi)
                print(f"  wrap {w}: GEOM {gname} -> body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)} (id {b})")
                bodies.append(b)
            else:
                print(f"  wrap {w}: type={wt} objid={oi} (unhandled)")
        bodies = sorted(set(bodies))
        print("  bodies:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies])
        lca = muscle_groups._lowest_common_ancestor(m, bodies)
        print("  LCA:", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, lca) if lca else "world")
        for b in bodies:
            print("    path", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), "->",
                  [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, x) for x in muscle_groups._path_to_root(m, b)])
        jids, _, _ = muscle_groups.crossed_joints_of(m, t)
        print("  crossed joints:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids])

    # 统计 crossed_joints 为空的肌肉数量
    mm = muscle_groups.build_map(m)
    empty = [e.name for e in mm.entries if not e.crossed_joints]
    print(f"\n未跨关节的肌肉数: {len(empty)} / {len(mm)}")
    print("前 20:", empty[:20])
    from collections import Counter
    print("按来源文件:", Counter(e.source_file for e in mm.entries if not e.crossed_joints))
    print("按权威分组:", Counter(e.limb_group for e in mm.entries if not e.crossed_joints))


if __name__ == "__main__":
    main()
