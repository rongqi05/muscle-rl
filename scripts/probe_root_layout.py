"""实证确定根节点 qpos/qvel 布局（关节名顺序 ≠ qpos 槽位顺序的可能性）。

用法::

    PYTHONPATH=. python scripts/probe_root_layout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402


def main() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(paths.MODEL_XML))
    pelvis = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))

    print("nq =", model.nq, " nv =", model.nv)
    print("\n=== root joints: jnt_id, name, type, qposadr, dofadr, axis ===")
    adr0 = int(model.body_jntadr[pelvis])
    num0 = int(model.body_jntnum[pelvis])
    for j in range(adr0, adr0 + num0):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(
            f"  jnt {j}: {name:16s} type={mujoco.mjtJoint(int(model.jnt_type[j])).name:10s} "
            f"qposadr={int(model.jnt_qposadr[j])} dofadr={int(model.jnt_dofadr[j])} "
            f"axis={np.round(np.asarray(model.jnt_axis[j]), 3)}"
        )

    ref = np.zeros(model.nq)
    data = mujoco.MjData(model)
    data.qpos[:] = ref
    mujoco.mj_forward(model, data)
    base = np.asarray(data.xpos[pelvis]).copy()
    print(f"\n=== 基准: qpos 全零 -> pelvis 世界位置 {np.round(base, 4)} ===")

    print("\n=== 逐个 qpos 槽位 +0.2 后的 pelvis 世界位置 ===")
    rows = []
    for k in range(6):
        d = mujoco.MjData(model)
        d.qpos[:] = ref
        d.qpos[k] = 0.2
        mujoco.mj_forward(model, d)
        pos = np.asarray(d.xpos[pelvis]).copy()
        rows.append({"qpos_idx": k, "delta": (pos - base).tolist(), "pos": pos.tolist()})
        print(f"  qpos[{k}] = 0.2 -> pelvis Δ = {np.round(pos - base, 4)}")

    print("\n=== 逐个 qvel 槽位 +1.0 后的 pelvis 线/角速度 ===")
    for k in range(6):
        d = mujoco.MjData(model)
        d.qpos[:] = ref
        d.qvel[k] = 1.0
        mujoco.mj_forward(model, d)
        # 由 mj_forward 后的 cvel/qvel 关系推断：直接用 jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, d, jacp, jacr, pelvis)
        j = int(model.jnt_dofadr[adr0 + min(k, num0 - 1)]) if k < num0 else k
        print(f"  dof({k}) 平移雅可比 = {np.round(jacp[:, k], 4)}  旋转雅可比 = {np.round(jacr[:, k], 4)}")

    out = {
        "nq": int(model.nq),
        "root_joint_table": [
            {
                "jnt_id": j,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                "qposadr": int(model.jnt_qposadr[j]),
                "dofadr": int(model.jnt_dofadr[j]),
                "axis": np.asarray(model.jnt_axis[j]).tolist(),
            }
            for j in range(adr0, adr0 + num0)
        ],
        "qpos_slot_effect": rows,
    }
    p = Path("reports/root_layout.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
