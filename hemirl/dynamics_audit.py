"""动力学闭环审计：策略 → 激励 → 激活 → 肌力 → 广义力矩 → MuJoCo 动力学 → 观测。

对应任务书第五节。审计分三层：

**A. 静态代码审计**（读上游源码，给出文件与行号）
    确认 ``step`` 中除了 ``kinematic_play`` 分支外没有对 ``qpos`` 的覆盖；
    确认 ``do_simulation`` 只是把动作写进 ``data.ctrl`` 然后 ``mj_step``；
    确认根节点只在 ``reset_model`` 里由 ``set_state`` 初始化。

**B. 模型结构审计**
    执行器是否全部是 muscle、有无额外 motor；有无 freejoint；
    equality 约束的类型与对象；被动项（damping/armature/stiffness/frictionloss）；
    参与接触的几何与摩擦。

**C. 运行时审计**（真实 rollout）
    * 零肌肉激励时躯体应因重力下落（证明是动力学而非回放）；
    * ``qfrc_applied`` / ``xfrc_applied`` 全程为 0（无人为外力/残差力矩）；
    * 状态不被参考轨迹覆盖（与 ``qpos_ref`` 的偏差随时间增长）；
    * 各力项范数分解（肌肉/被动/约束/重力）；
    * 闭环数值验证：``actuator_force == mju_muscleGain*act + mju_muscleBias``，
      且 ``act`` 由 ``ctrl`` 经激活动力学演化（act ≠ ctrl）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from hemirl import paths
from hemirl.forces import decompose_constraint_forces, muscle_active_passive_split
from hemirl.rollout import root_state


# ------------------------------------------------------------------ A 静态


def static_code_audit() -> Dict[str, Any]:
    """读取上游 msgym 源码，定位关键行为。"""
    env_file = paths.MSGYM_ROOT / "msgym" / "envs" / "locomotionFull_v1.py"
    utils_file = paths.MSGYM_ROOT / "msgym" / "envs" / "utils.py"
    gym_file = Path(__file__).resolve().parent  # placeholder, 下面单独查找

    src = env_file.read_text(encoding="utf-8")
    lines = src.splitlines()

    out: Dict[str, Any] = {"env_file": str(env_file), "utils_file": str(utils_file)}

    # 1) kinematic_play 的默认值
    m = re.search(r"kinematic_play:\s*bool\s*=\s*(\w+)", src)
    out["kinematic_play_default"] = m.group(1) if m else None

    # 2) 找出 step() 内所有对真实状态(self.data.qpos/qvel/qacc)的赋值
    step_start = src.find("def step(")
    step_end = src.find("def reset_model(")
    step_src = src[step_start:step_end]
    state_assigns = []
    ref_assigns = []
    for offset, ln in enumerate(step_src.splitlines()):
        line_no = src[:step_start].count("\n") + 1 + offset
        stripped = ln.strip()
        if re.search(r"self\.data\.(qpos|qvel|qacc)\s*\[", stripped) and "==" not in stripped:
            state_assigns.append({"line": line_no, "code": stripped})
        elif re.search(r"self\.qpos_ref(_future)?\s*[:=]", stripped):
            ref_assigns.append({"line": line_no, "code": stripped})
    out["step_state_assignments"] = state_assigns
    out["step_reference_buffer_assignments"] = ref_assigns
    out["step_lines"] = {
        "start": src[:step_start].count("\n") + 1,
        "end": src[:step_end].count("\n") + 1,
    }

    # 3) kinematic_play 的条件行号
    kp_lines = [i + 1 for i, ln in enumerate(lines) if "kinematic_play" in ln]
    out["kinematic_play_lines"] = kp_lines

    # 4) reset_model 里的 set_state
    rm_start = src.find("def reset_model(")
    rm_end = src.find("def _get_qpos_reward(")
    rm_src = src[rm_start:rm_end]
    out["reset_model_calls_set_state"] = "self.set_state(" in rm_src
    out["reset_model_line"] = src[:rm_start].count("\n") + 1

    # 5) 是否存在 PD 控制器 / 额外力矩
    pd_patterns = ["qfrc_applied", "xfrc_applied", "kp", "kd", "actuator", "position_actuator"]
    out["pd_or_extra_torque_patterns"] = {
        p: bool(re.search(rf"\b{p}\b", src)) for p in pd_patterns
    }

    # 6) gymnasium do_simulation / _step_mujoco_simulation
    import gymnasium
    from gymnasium.envs.mujoco import mujoco_env

    gym_src = Path(mujoco_env.__file__).read_text(encoding="utf-8")
    g_lines = gym_src.splitlines()
    out["gymnasium_mujoco_env_file"] = str(Path(mujoco_env.__file__))
    _step_idx = gym_src.find("def _step_mujoco_simulation")
    out["gymnasium_step_mujoco_simulation_line"] = gym_src[:_step_idx].count("\n") + 1
    out["gymnasium_step_mujoco_simulation_body"] = [
        l.strip()
        for l in gym_src[_step_idx : gym_src.find("def render(")].splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    _ds_idx = gym_src.find("def do_simulation")
    out["gymnasium_do_simulation_line"] = gym_src[:_ds_idx].count("\n") + 1
    out["gymnasium_version"] = gymnasium.__version__

    return out


# ------------------------------------------------------------------ B 模型


def model_structure_audit(model) -> Dict[str, Any]:
    import mujoco

    nu = int(model.nu)
    dyntype = np.asarray(model.actuator_dyntype)
    gaintype = np.asarray(model.actuator_gaintype)
    biastype = np.asarray(model.actuator_biastype)
    trntype = np.asarray(model.actuator_trntype)

    names = lambda obj, i: mujoco.mj_id2name(model, obj, i)

    joint_types = {}
    for j in range(model.njnt):
        t = mujoco.mjtJoint(int(model.jnt_type[j])).name
        joint_types[t] = joint_types.get(t, 0) + 1

    eq_types: Dict[str, int] = {}
    eq_coupled_to: Dict[str, int] = {}
    for e in range(model.neq):
        t = mujoco.mjtEq(int(model.eq_type[e])).name
        eq_types[t] = eq_types.get(t, 0) + 1
        obj2 = names(mujoco.mjtObj.mjOBJ_JOINT, int(model.eq_obj2id[e])) or "?"
        eq_coupled_to[obj2] = eq_coupled_to.get(obj2, 0) + 1

    contact_geoms = []
    contype = np.asarray(model.geom_contype)
    conaff = np.asarray(model.geom_conaffinity)
    for g in np.where((contype != 0) | (conaff != 0))[0]:
        contact_geoms.append(
            {
                "geom": names(mujoco.mjtObj.mjOBJ_GEOM, int(g)),
                "body": names(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])),
                "friction": [float(v) for v in np.asarray(model.geom_friction)[g][:3]],
                "solref": [float(v) for v in np.asarray(model.geom_solref)[g][:2]],
            }
        )

    def summarize(arr, label):
        nz = np.asarray(arr)[np.asarray(arr) != 0]
        return {
            "label": label,
            "n_total": int(np.asarray(arr).size),
            "n_nonzero": int(nz.size),
            "unique": sorted({round(float(v), 6) for v in nz}),
        }

    disable_flags = int(model.opt.disableflags)
    known_flags = [
        "mjDSBL_CONSTRAINT",
        "mjDSBL_EQUALITY",
        "mjDSBL_FRICTIONLOSS",
        "mjDSBL_LIMIT",
        "mjDSBL_CONTACT",
        "mjDSBL_PASSIVE",
        "mjDSBL_AUTORESET",
        "mjDSBL_CLAMPCTRL",
        "mjDSBL_WARMSTART",
        "mjDSBL_FILTERPARENT",
        "mjDSBL_ACTUATION",
        "mjDSBL_REFSAFE",
        "mjDSBL_SENSOR",
        "mjDSBL_MIDPHASE",
        "mjDSBL_EULERDAMP",
        "mjDSBL_NATIVECCD",
        "mjDSBL_GRAVITY",
        "mjDSBL_DISABLED",
    ]
    disabled = [
        name
        for name in known_flags
        if hasattr(mujoco.mjtDisableBit, name)
        and (disable_flags & int(getattr(mujoco.mjtDisableBit, name).value))
    ]

    return {
        "counts": {
            "nbody": int(model.nbody),
            "njnt": int(model.njnt),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "na": int(model.na),
            "ngeom": int(model.ngeom),
            "ntendon": int(model.ntendon),
            "neq": int(model.neq),
            "nsensor": int(model.nsensor),
        },
        "actuators": {
            "all_muscle_dyntype": bool((dyntype == int(mujoco.mjtDyn.mjDYN_MUSCLE)).all()),
            "dyntype_values": sorted({int(v) for v in dyntype}),
            "gaintype_values": sorted({int(v) for v in gaintype}),
            "biastype_values": sorted({int(v) for v in biastype}),
            "transmission_types": sorted({int(v) for v in trntype}),
            "ctrlrange_unique": sorted({tuple(np.round(np.asarray(model.actuator_ctrlrange)[i], 6)) for i in range(min(nu, 700))}),
            "n_motor_actuators": int((dyntype != int(mujoco.mjtDyn.mjDYN_MUSCLE)).sum()),
        },
        "root": {
            "has_freejoint": any(
                int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) for j in range(model.njnt)
            ),
            "root_body": names(mujoco.mjtObj.mjOBJ_BODY, 1),
            "root_joints": [
                names(mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in range(int(model.body_jntadr[1]), int(model.body_jntadr[1]) + int(model.body_jntnum[1]))
            ],
        },
        "joint_types": joint_types,
        "equality": {
            "types": eq_types,
            "coupled_to": eq_coupled_to,
            "n_active": int(np.sum(np.asarray(model.eq_active0))) if hasattr(model, "eq_active0") else None,
        },
        "passive": {
            "dof_damping": summarize(model.dof_damping, "dof_damping"),
            "dof_armature": summarize(model.dof_armature, "dof_armature"),
            "dof_frictionloss": summarize(model.dof_frictionloss, "dof_frictionloss"),
            "jnt_stiffness": summarize(getattr(model, "jnt_stiffness", np.zeros(0)), "jnt_stiffness"),
        },
        "contact_geoms": contact_geoms,
        "opt": {
            "timestep": float(model.opt.timestep),
            "gravity": [float(v) for v in model.opt.gravity],
            "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name,
            "solver": mujoco.mjtSolver(int(model.opt.solver)).name,
            "iterations": int(model.opt.iterations),
            "disableflags": disable_flags,
            "disabled_features": disabled,
        },
    }


# ------------------------------------------------------------------ C 运行时


def runtime_audit(evaluator, n_steps: int = 60, zero_action: bool = False) -> Dict[str, Any]:
    """在真实环境上跑若干步，采集力项与闭环证据。"""
    import mujoco

    raw = evaluator.raw_env
    model = evaluator.model
    data = evaluator.data

    evaluator.scaler.reset()
    obs, _ = evaluator.env.reset(seed=0)
    rng = np.random.default_rng(0)

    rec: Dict[str, List[float]] = {
        "qfrc_applied_norm": [],
        "xfrc_applied_norm": [],
        "qfrc_actuator_norm": [],
        "qfrc_passive_norm": [],
        "qfrc_constraint_norm": [],
        "qfrc_bias_norm": [],
        "qacc_norm": [],
        "qvel_norm": [],
        "pelvis_z": [],
        "act_mean": [],
        "ctrl_mean": [],
        "actuator_force_mean_abs": [],
        "qpos_ref_dev": [],
        "ncon": [],
    }
    muscle_force_formula_err: List[float] = []
    pd_perturbation: List[float] = []
    root_vel_lin_diff: List[float] = []
    root_vel_ang_diff: List[float] = []
    constraint_groups: List[Dict[str, float]] = []

    lr_all = np.asarray(model.actuator_lengthrange)
    acc_all = np.asarray(model.actuator_acc0)
    gp = np.asarray(model.actuator_gainprm)[:, :9]
    bp = np.asarray(model.actuator_biasprm)[:, :9]

    for step in range(n_steps):
        if zero_action:
            action = np.zeros(evaluator.env.action_space.shape, dtype=np.float32)
        else:
            normalized = evaluator.stack.normalize_obs(obs)
            action = evaluator.stack.predict(normalized)
        obs, _r, terminated, truncated, _i = evaluator.env.step(action)

        length = np.asarray(data.actuator_length)
        vel = np.asarray(data.actuator_velocity)
        act = np.asarray(data.act)
        formula = np.zeros(model.nu)
        for a in (0, 1, 2, 100, 350, 699):
            gv = float(mujoco.mju_muscleGain(float(length[a]), float(vel[a]), lr_all[a], float(acc_all[a]), gp[a]))
            bv = float(mujoco.mju_muscleBias(float(length[a]), lr_all[a], float(acc_all[a]), bp[a]))
            formula[a] = gv * act[a] + bv
        model_force = np.asarray(data.actuator_force)
        errs = [abs(formula[a] - float(model_force[a])) for a in (0, 1, 2, 100, 350, 699)]
        muscle_force_formula_err.append(float(max(errs)))
        rec["qfrc_applied_norm"].append(float(np.linalg.norm(data.qfrc_applied)))
        rec["xfrc_applied_norm"].append(float(np.linalg.norm(np.asarray(data.xfrc_applied))))
        rec["qfrc_actuator_norm"].append(float(np.linalg.norm(data.qfrc_actuator)))
        rec["qfrc_passive_norm"].append(float(np.linalg.norm(data.qfrc_passive)))
        rec["qfrc_constraint_norm"].append(float(np.linalg.norm(data.qfrc_constraint)))
        rec["qfrc_bias_norm"].append(float(np.linalg.norm(data.qfrc_bias)))
        rec["qacc_norm"].append(float(np.linalg.norm(data.qacc)))
        rec["qvel_norm"].append(float(np.linalg.norm(data.qvel)))
        rec["pelvis_z"].append(float(data.xpos[evaluator.pelvis_id][2]))
        rec["act_mean"].append(float(np.mean(act)))
        rec["ctrl_mean"].append(float(np.mean(np.asarray(data.ctrl))))
        rec["actuator_force_mean_abs"].append(float(np.mean(np.abs(model_force))))
        qref = np.asarray(raw.qpos_ref)
        rec["qpos_ref_dev"].append(float(np.mean(np.abs(np.asarray(data.qpos)[3:] - qref[3:]))))
        rec["ncon"].append(float(data.ncon))

        # 参考轨迹下一次是否会覆盖当前状态：把当前状态与参考比较
        pd_perturbation.append(float(np.linalg.norm(np.asarray(data.qpos)[:3] - qref[:3])))

        # 根节点速度：Jacobian 写法的世界系角速度 vs 旧式「槽位重排」写法
        rs = root_state(model, data, evaluator.pelvis_id)
        root_vel_lin_diff.append(float(np.max(np.abs(rs["lin_vel"] - rs["lin_vel_index_formula"]))))
        root_vel_ang_diff.append(float(np.max(np.abs(rs["ang_vel"] - rs["ang_vel_index_formula"]))))

        # 约束力按类型分解（不把 qfrc_constraint 整体归因 equality）
        constraint_groups.append(decompose_constraint_forces(model, data)["per_group_norm"])
        if terminated or truncated:
            break

    out: Dict[str, Any] = {
        "n_steps": len(rec["pelvis_z"]),
        "policy_action_zeroed": zero_action,
        "note": (
            "zero_action=True 表示把**策略动作**置零；经 MuscleNormWrapper 后 "
            "excitation ≈ 0.076 而不是 0（动作 0 映射到 sigmoid 的中间偏低处）"
        ),
        "qfrc_applied_max": max(rec["qfrc_applied_norm"]),
        "xfrc_applied_max": max(rec["xfrc_applied_norm"]),
        "qfrc_actuator_norm_mean": float(np.mean(rec["qfrc_actuator_norm"])),
        "qfrc_passive_norm_mean": float(np.mean(rec["qfrc_passive_norm"])),
        "qfrc_constraint_norm_mean": float(np.mean(rec["qfrc_constraint_norm"])),
        "qfrc_bias_norm_mean": float(np.mean(rec["qfrc_bias_norm"])),
        "qacc_norm_mean": float(np.mean(rec["qacc_norm"])),
        "pelvis_z_start": rec["pelvis_z"][0],
        "pelvis_z_end": rec["pelvis_z"][-1],
        "pelvis_z_monotone_decreasing": bool(
            all(b <= a + 1e-6 for a, b in zip(rec["pelvis_z"], rec["pelvis_z"][1:]))
        ),
        "ctrl_mean_mean": float(np.mean(rec["ctrl_mean"])),
        "act_mean_mean": float(np.mean(rec["act_mean"])),
        "act_equals_ctrl": bool(np.allclose(np.asarray(rec["act_mean"]), np.asarray(rec["ctrl_mean"]), atol=1e-9)),
        "actuator_force_mean_abs_mean": float(np.mean(rec["actuator_force_mean_abs"])),
        "muscle_force_formula_max_abs_err_sampled": float(np.max(muscle_force_formula_err)),
        "qpos_ref_dev_first": rec["qpos_ref_dev"][0],
        "qpos_ref_dev_last": rec["qpos_ref_dev"][-1],
        "root_pos_ref_dist_first": pd_perturbation[0],
        "root_pos_ref_dist_last": pd_perturbation[-1],
        "ncon_mean": float(np.mean(rec["ncon"])),
        "state_is_not_overwritten_by_reference": bool(
            rec["qpos_ref_dev"][-1] > 1e-9 and rec["qpos_ref_dev"][-1] != rec["qpos_ref_dev"][0]
        ),
        # 根速度：Jacobian 与旧写法的偏差（旧写法仅 lin 分量的槽位顺序恰好正确）
        "root_velocity": {
            "method": "mujoco.mj_jacBody(qpos) @ qvel",
            "frame": "world",
            "lin_vel_index_formula_max_abs_diff": float(np.max(root_vel_lin_diff)) if root_vel_lin_diff else None,
            "ang_vel_index_formula_max_abs_diff": float(np.max(root_vel_ang_diff)) if root_vel_ang_diff else None,
            "note": (
                "线速度的旧槽位写法在该模型上恰好等价；角速度**不等价**，"
                "因此一律改用旋转 Jacobian（与模型结构无关，不会随关节顺序变化而静默失效）"
            ),
        },
        # 约束力分解：接触 / 限位 / equality / 摩擦各自的实际贡献
        "constraint_decomposition": {
            "per_group_norm_mean": _mean_of_dicts(constraint_groups),
            "n_steps": len(constraint_groups),
            "note": (
                "qfrc_constraint 是接触 + 限位 + equality + 摩擦的**合计**；"
                "本项按 mjCNSTR_* 类型拆开，因此可以判断到底是谁在受力，"
                "而不是把整体范数归因于 equality。"
            ),
        },
        "series": {k: [float(v) for v in vals] for k, vals in rec.items()},
    }

    # 同状态一致性检验：用当前 qpos/qvel/act/ctrl 重新 mj_forward，使长度/速度/激活/力
    # 取自同一状态，此时「mju_muscleGain*act + mju_muscleBias == data.actuator_force」
    # 应逐元素精确成立。运动轨迹上的采样误差来自半隐式欧拉（mj_step 先用步前速度算力再积分）。
    d3 = mujoco.MjData(model)
    d3.qpos[:] = np.asarray(data.qpos)
    d3.qvel[:] = np.asarray(data.qvel)
    d3.act[:] = np.asarray(data.act)
    d3.ctrl[:] = np.asarray(data.ctrl)
    mujoco.mj_forward(model, d3)
    l3 = np.asarray(d3.actuator_length)
    v3 = np.asarray(d3.actuator_velocity)
    a3 = np.asarray(d3.act)
    f3 = np.asarray(d3.actuator_force)
    g3 = np.zeros(model.nu)
    b3 = np.zeros(model.nu)
    for a in range(model.nu):
        g3[a] = float(mujoco.mju_muscleGain(float(l3[a]), float(v3[a]), lr_all[a], float(acc_all[a]), gp[a]))
        b3[a] = float(mujoco.mju_muscleBias(float(l3[a]), lr_all[a], float(acc_all[a]), bp[a]))
    err3 = np.abs(g3 * a3 + b3 - f3)
    out["same_state_formula_check"] = {
        "max_abs_err": float(err3.max()),
        "mean_abs_err": float(err3.mean()),
        "n_nonzero_velocity": int((np.abs(v3) > 1e-9).sum()),
        "max_abs_velocity": float(np.abs(v3).max()),
        "exact": bool(err3.max() == 0.0),
    }

    # 肌肉力拆分：F_active = F(act) - F(act=0)，与肌力缩放验证使用同一套定义
    out["muscle_active_passive_split"] = muscle_active_passive_split(model, data)

    # 终止元数据：每一步（含终止步）都被记录，运行时间来自实际仿真时间差
    out["termination_metadata"] = evaluator.env.termination_metadata()
    out["accounting"] = evaluator.env.accounting_report()
    return out


def _mean_of_dicts(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """对一组同键字典逐键取均值。"""
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r})
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


__all__ = ["static_code_audit", "model_structure_audit", "runtime_audit"]
