"""模型内省：核实 MS-Human-700 的自由度、执行器、肌力字段与被动参数。

用法::

    PYTHONPATH=. python scripts/inspect_model.py
    PYTHONPATH=. python scripts/inspect_model.py --xml external/MS-Human-700/MS-Human-700-Locomotion.xml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402


def _safe(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def _summarize_joint_damping(model) -> Dict[str, Any]:
    """汇总被动关节参数。

    注意：MuJoCo 3.11 起 `dof_stiffness` 已移除，关节刚度位于 `jnt_stiffness`。
    """
    out: Dict[str, Any] = {}
    arrays = {
        "damping": np.asarray(model.dof_damping),
        "armature": np.asarray(model.dof_armature),
        "frictionloss": np.asarray(model.dof_frictionloss),
    }
    stiff = getattr(model, "jnt_stiffness", None)
    if stiff is not None:
        arrays["stiffness(jnt)"] = np.asarray(stiff)
    for label, arr in arrays.items():
        nz = arr[arr != 0]
        out[label] = {
            "len": int(arr.size),
            "n_nonzero": int(nz.size),
            "min": float(nz.min()) if nz.size else 0.0,
            "max": float(nz.max()) if nz.size else 0.0,
            "unique_head": sorted({round(float(v), 6) for v in nz})[:8],
        }
    return out


def _actuator_report(model, xml_path: Path) -> Dict[str, Any]:
    import mujoco

    nu = model.nu
    dyntype = np.asarray(model.actuator_dyntype)
    gaintype = np.asarray(model.actuator_gaintype)
    biastype = np.asarray(model.actuator_biastype)
    gainprm = np.asarray(model.actuator_gainprm)
    biasprm = np.asarray(model.actuator_biasprm)
    ctrlrange = np.asarray(model.actuator_ctrlrange)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(nu)]
    trnid = np.asarray(model.actuator_trnid)
    gear = np.asarray(model.actuator_gear)

    # 判断 F0 候选字段：对 muscle dyntype 的 actuator，比较 gainprm[:,0] 与其它候选
    is_muscle = dyntype == int(mujoco.mjtDyn.mjDYN_MUSCLE)
    report: Dict[str, Any] = {
        "nu": int(nu),
        "n_muscle_dyntype": int(is_muscle.sum()),
        "dyntype_unique": sorted({int(v) for v in dyntype}),
        "gaintype_unique": sorted({int(v) for v in gaintype}),
        "biastype_unique": sorted({int(v) for v in biastype}),
        "gainprm_nonzero_cols": [int(c) for c in np.where((gainprm != 0).any(axis=0))[0]],
        "biasprm_nonzero_cols": [int(c) for c in np.where((biasprm != 0).any(axis=0))[0]],
        "ctrlrange_unique": sorted({tuple(np.round(r, 6)) for r in ctrlrange}),
        "gear_nonzero_cols": [int(c) for c in np.where((gear != 0).any(axis=0))[0]],
        "gear_col0_unique": sorted({float(v) for v in gear[:, 0]})[:8],
    }
    if is_muscle.any():
        f0 = gainprm[is_muscle, 0]
        report["F0_from_gainprm_col0"] = {
            "min": float(f0.min()),
            "max": float(f0.max()),
            "mean": float(f0.mean()),
            "head5": [float(v) for v in f0[:5]],
        }
        # 旧版本用 sex/force 存于 actuator_gear? 这里给出对照
        report["gear_col0_for_muscle"] = {
            "min": float(gear[is_muscle, 0].min()),
            "max": float(gear[is_muscle, 0].max()),
        }
    report["sample_actuators"] = [
        {
            "id": i,
            "name": names[i],
            "dyntype": int(dyntype[i]),
            "gainprm[:3]": [float(v) for v in gainprm[i, :3]],
            "biasprm[:3]": [float(v) for v in biasprm[i, :3]],
            "ctrlrange": [float(v) for v in ctrlrange[i]],
            "trnid": [int(v) for v in trnid[i]],
            "gear[:2]": [float(v) for v in gear[i, :2]],
        }
        for i in range(min(3, nu))
    ]
    report["n_unique_names"] = len(set(names))
    return report


def collect(xml_path: Path) -> Dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # 根节点：world 的直接子 body 及其 freejoint
    root_bodies = []
    for b in range(1, model.nbody):
        if model.body_parentid[b] == 0:
            jadr = model.body_jntadr[b]
            jnum = model.body_jntnum[b]
            joints = []
            for j in range(jadr, jadr + jnum):
                joints.append(
                    {
                        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                        "type": int(model.jnt_type[j]),
                        "nq": int(model.jnt_type[j]) if False else None,
                    }
                )
            root_bodies.append(
                {
                    "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                    "njoint": int(jnum),
                    "joints": joints,
                }
            )

    freejoints = [
        {
            "id": j,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])),
        }
        for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]

    # 关节类型统计
    type_counts: Dict[str, int] = {}
    for j in range(model.njnt):
        t = mujoco.mjtJoint(int(model.jnt_type[j])).name
        type_counts[t] = type_counts.get(t, 0) + 1

    # 接触几何
    contype = np.asarray(model.geom_contype)
    conaffinity = np.asarray(model.geom_conaffinity)
    n_colliding = int(((contype != 0) | (conaffinity != 0)).sum())

    # equality 约束
    eq_types: Dict[str, int] = {}
    for e in range(model.neq):
        t = mujoco.mjtEq(int(model.eq_type[e])).name
        eq_types[t] = eq_types.get(t, 0) + 1

    out: Dict[str, Any] = {
        "xml": str(xml_path),
        "model_name": model.names[:0] or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MODEL, 0),
        "counts": {
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "na": int(model.na),
            "ngeom": int(model.ngeom),
            "nsite": int(model.nsite),
            "ntendon": int(model.ntendon),
            "neq": int(model.neq),
            "nsensor": int(model.nsensor),
            "nkey": int(model.nkey),
        },
        "opt": {
            "timestep": float(model.opt.timestep),
            "gravity": [float(v) for v in model.opt.gravity],
            "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name,
            "solver": mujoco.mjtSolver(int(model.opt.solver)).name,
            "cone": mujoco.mjtCone(int(model.opt.cone)).name,
            "iterations": int(model.opt.iterations),
            "disableflags": int(model.opt.disableflags),
            "enableflags": int(model.opt.enableflags),
        },
        "joint_type_counts": type_counts,
        "root_bodies": root_bodies,
        "freejoints": freejoints,
        "geom_collision": {
            "n_geoms_with_contact": n_colliding,
            "n_geoms": int(model.ngeom),
        },
        "equality_types": eq_types,
        "passive_dof_params": _summarize_joint_damping(model),
        "actuators": _actuator_report(model, xml_path),
    }

    # 关键 body 是否存在
    key_bodies = ["pelvis", "sternum", "head_neck", "toes_r", "toes_l", "proximal_row_r", "proximal_row_l"]
    out["key_bodies"] = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) for name in key_bodies
    }

    # 初始 key_frame 检查
    if model.nkey > 0:
        qpos0 = np.asarray(model.key_qpos[0])
        out["keyframe0"] = {
            "qpos_norm": float(np.linalg.norm(qpos0)),
            "qpos_head": [float(v) for v in qpos0[:7]],
            "qpos_first_actuated": [float(v) for v in qpos0[7:12]],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MS-Human-700 模型内省")
    parser.add_argument("--xml", type=str, default=str(paths.MODEL_XML))
    parser.add_argument("--out", type=str, default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    xml = Path(args.xml).resolve()
    info = collect(xml)
    text = json.dumps(info, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
