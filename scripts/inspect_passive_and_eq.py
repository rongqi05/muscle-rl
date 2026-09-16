"""检查被动关节参数与 equality 约束的分布（动力学审计所需）。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hemirl import paths  # noqa: E402


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(paths.MODEL_XML))
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]

    stiff = np.asarray(m.jnt_stiffness)
    print("=== joints with stiffness != 0 ===")
    for j in np.where(stiff != 0)[0]:
        print(f"  {jn[j]:40s} stiffness={stiff[j]}  body={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j]))}")

    print("\n=== damping distribution (nonzero) ===")
    damp = np.asarray(m.dof_damping)
    for j in np.where(damp != 0)[0]:
        print(f"  {jn[j]:40s} damping={damp[j]:<8g} armature={np.asarray(m.dof_armature)[j]:<8g}")

    print("\n=== equality constraints ===")
    for e in range(m.neq):
        t = mujoco.mjtEq(int(m.eq_type[e])).name
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e)
        obj1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.eq_obj1id[e]))
        obj2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.eq_obj2id[e]))
        active = int(np.asarray(m.eq_active0)[e]) if hasattr(m, "eq_active0") else "?"
        print(f"  [{e:2d}] type={t:12s} name={name} obj1={obj1} obj2={obj2} "
              f"data={[round(float(v),6) for v in np.asarray(m.eq_data)[e][:5]]} "
              f"active={active} "
              f"solref={[round(float(v),4) for v in np.asarray(m.eq_solref)[e]]}")

    print("\n=== joint types ===")
    print(Counter(mujoco.mjtJoint(int(m.jnt_type[j])).name for j in range(m.njnt)))

    print("\n=== geoms with contact ===")
    contype = np.asarray(m.geom_contype)
    conaff = np.asarray(m.geom_conaffinity)
    idx = np.where((contype != 0) | (conaff != 0))[0]
    for g in idx:
        print(f"  {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g):28s} body={mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])):16s} "
              f"contype={contype[g]} conaffinity={conaff[g]} friction={[round(float(v),3) for v in np.asarray(m.geom_friction)[g]]} "
              f"solref={[round(float(v),4) for v in np.asarray(m.geom_solref)[g]]} priority={int(m.geom_priority[g])}")

    print("\n=== option flags ===")
    print("disableflags:", int(m.opt.disableflags), "enableflags:", int(m.opt.enableflags))
    for name in ["mjDSBL_CONTACT", "mjDSBL_CONSTRAINT", "mjDSBL_PASSIVE", "mjDSBL_GRAVITY",
                 "mjDSBL_ACTUATION", "mjDSBL_FRICTIONLOSS", "mjDSBL_LIMIT", "mjDSBL_DAMPER"]:
        print(f"  {name} = {getattr(mujoco.mjtDisableBit, name).value}")

    print("\n=== keyframe qpos ===")
    print("qpos0:", np.asarray(m.qpos0))
    print("key_qpos[0]:", np.asarray(m.key_qpos[0]))
    print("key_ctrl[0]:", np.asarray(m.key_ctrl[0])[:10], "...")
    print("gravity:", np.asarray(m.opt.gravity))


if __name__ == "__main__":
    main()
