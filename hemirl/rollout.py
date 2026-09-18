"""评估运行器：固定策略权重下的动力学 rollout，记录完整证据链。

## 数据流（全部落盘可查）

::

    策略输出 a_policy ∈ [-1,1]^700                （hemirl.policy）
      → MuscleNormWrapper: a_env = 1/(1+exp(-5(a-0.5)))  ∈ (0,1)^700
      → env.step 内部: data.ctrl[:] = a_env        （gymnasium MujocoEnv._step_mujoco_simulation）
      → MuJoCo muscle activation 动力学: data.act
      → 肌肉力: data.actuator_force
      → 广义肌肉力矩: data.qfrc_actuator
      → 接触/约束/被动/重力: mj_step 求解
      → 下一时刻状态 data.qpos/qvel/qacc
      → 观测（含 qacc / actuator_force / act）→ 下一步策略输入

## 本模块的几条硬性口径

1. **根节点速度用旋转 Jacobian**（``omega_world = J_rot(q) @ qvel``），
   不再使用「按槽位重排」的写法。旧写法只在特定 joint 顺序下成立，
   一旦上游改模型就会静默出错；Jacobian 写法则与模型结构无关。
   旧写法的结果仍会被计算并记录差异（``jacobian_vs_index_max_diff``），作为交叉验证。
2. **运行时间用实际仿真时间差**（``data.time`` 的差），不是「步数 × 标称 dt」。
   两者都会被记录，差值落在 ``accounting`` 里。
3. **每一步都记录，含触发终止的最后一步**：终止判据在 ``env.step`` 之后求值并写入 ledger，
   不存在「先 break 再补记录」的窗口。
4. **参考跟踪误差是描述性指标**，不作为终止依据，也不代表平衡能力。
5. 终止语义由 :class:`hemirl.research_env.ResearchLocomotionEnv` 统一给出
   （``terminated`` = 物理跌倒/数值异常；``truncated`` = 达到规定评估时长）。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hemirl import muscle_groups, paths, provenance
from hemirl.envs import OfficialEnvConfig
from hemirl.forces import (
    decompose_constraint_forces,
    generalized_force_terms,
    muscle_active_passive_split,
)
from hemirl.muscle_actuator import AppliedStrength, StrengthScaler, StrengthSpec
from hemirl.research_env import (
    REWARD_MODE_OFFICIAL,
    ResearchEnvConfig,
    build_research_env,
)
from hemirl.termination import TerminationConfig

#: 分组键固定顺序，方便落盘对比
GROUP_KEYS = ("L/upper", "R/upper", "L/lower", "R/lower", "M/torso", "L/torso", "R/torso")


# ------------------------------------------------------------------ 状态与接触


def root_state(model, data, pelvis_id: Optional[int] = None) -> Dict[str, Any]:
    """根节点（骨盆）位姿与速度，**速度一律由 Jacobian 得到**。

    定义与坐标约定：

    * ``pos``：骨盆 body 原点在**世界系**的位置（m）；
    * ``rot``：骨盆 body 到世界系的旋转矩阵；
    * ``lin_vel``：骨盆 body 原点的世界系线速度（m/s），``J_pos(q) @ qvel``；
    * ``ang_vel``：骨盆 body 的**世界系**角速度（rad/s），``J_rot(q) @ qvel``；
    * 采样时刻：``data`` 当前状态（本模块在每次 ``env.step`` 之后立即调用）。

    实现：``mujoco.mj_jacBody`` 给出 ``J_pos`` / ``J_rot``（3×nv），
    再左乘 ``qvel``。这与自由关节的四元数索引、以及任何人手写的槽位顺序假设**都无关**，
    因此模型改动不会让它静默失效。

    为便于回归，同时给出旧式「槽位重排」结果（``lin_vel_index_formula`` /
    ``ang_vel_index_formula``）；两者之差由调用方汇总。
    """
    import mujoco

    if pelvis_id is None:
        pelvis_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    nv = int(model.nv)
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, int(pelvis_id))
    qvel = np.asarray(data.qvel, dtype=float)

    lin_vel = jacp @ qvel
    ang_vel = jacr @ qvel

    # 旧式槽位重排（仅供回归对照；依赖 pelvis_tz/ty/tx 与 tilt/list/rotation 的顺序）
    lin_index = np.array([qvel[2], -qvel[0], qvel[1]])
    ang_index = np.array([qvel[4], -qvel[3], qvel[5]])

    return {
        "pos": np.asarray(data.xpos[pelvis_id], dtype=float).copy(),
        "rot": np.asarray(data.xmat[pelvis_id], dtype=float).reshape(3, 3).copy(),
        "lin_vel": lin_vel,
        "ang_vel": ang_vel,
        "lin_vel_index_formula": lin_index,
        "ang_vel_index_formula": ang_index,
        "velocity_method": "mujoco.mj_jacBody(qpos) @ qvel",
        "lin_vel_frame": "world",
        "ang_vel_frame": "world",
        "units": {"lin_vel": "m/s", "ang_vel": "rad/s", "pos": "m"},
    }


def contact_metrics(model, data) -> Dict[str, float]:
    """接触指标：接触点数、法向力合计、各足接触与足底合力。

    法向力由 ``mj_contactForce`` 读取（前 3 维为接触坐标系下的力，第 0 个分量是法向力）。
    """
    import mujoco

    ncon = int(data.ncon)
    out: Dict[str, float] = {"ncon": float(ncon), "total_normal_force": 0.0}
    foot_names = ("calcn_r", "calcn_l", "toes_r", "toes_l")
    foot_force = {n: 0.0 for n in foot_names}
    total = 0.0
    buf = np.zeros(6, dtype=np.float64)
    for i in range(ncon):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, buf)
        f_normal = float(abs(buf[0]))
        total += f_normal
        for gid in (con.geom1, con.geom2):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
            if bname in foot_force:
                foot_force[bname] += f_normal
    out["total_normal_force"] = total
    for n in foot_names:
        out[f"normal_force_{n}"] = foot_force[n]
    out["any_foot_contact"] = float(any(foot_force[n] > 1e-6 for n in foot_names))
    out["n_contact_geoms"] = float(len(np.unique([data.contact[i].geom1 for i in range(ncon)])))
    return out


# ------------------------------------------------------------------ 结果结构


@dataclass
class EpisodeResult:
    """一次 episode 的汇总指标。"""

    label: str
    seed: int
    strength: Dict[str, Any]

    # 计时（实际仿真时间差，不是步数×dt）
    steps: int
    n_physics_steps: int
    control_dt_s: float
    sim_time_s: float
    sim_time_nominal_s: float
    wall_time_s: float

    # 终止语义（三分 + 来源）
    terminated: bool
    truncated: bool
    termination_reason: Optional[str]
    termination_source: Optional[str]
    is_numeric_anomaly: bool
    is_physical_fall: bool
    n_official_term_flags: int
    n_steps_after_official_term: int
    reached_time_limit: bool
    reached_step_cap: bool

    # 根节点与整体运动
    root_start_pos: List[float]
    root_end_pos: List[float]
    displacement_xy: float
    forward_displacement_x: float
    lateral_drift_y: float
    distance_3d: float
    mean_speed_xy: float
    forward_speed_x: float
    ref_forward_speed: float
    mean_pelvis_height: float
    min_pelvis_height: float
    final_pelvis_height: float
    max_pelvis_up_tilt_deg: float
    final_pelvis_up_tilt_deg: float
    posture_change_deg: float
    max_root_speed: float
    max_root_ang_speed: float
    final_root_lin_vel: List[float]
    final_root_ang_vel: List[float]

    # 参考跟踪（**描述性**，不作终止依据，不代表平衡能力）
    mean_qpos_track_err: float
    max_qpos_track_err: float
    final_qpos_track_err: float

    # 肌肉 / 动作（分组统计）
    group_stats: Dict[str, Dict[str, float]]
    action_abs_mean: float
    action_abs_max: float
    excitation_mean: float
    activation_mean: float
    actuator_force_abs_mean: float

    # 奖励拆分（用于核查「物理存活」与「模仿」是否被混在一起）
    reward_totals: Dict[str, float]

    # 接触
    ncon_mean: float
    ncon_max: float
    total_normal_force_mean: float
    foot_contact_fraction: float

    # 数值健康度
    max_abs_qvel: float
    max_abs_qacc: float
    qfrc_applied_abs_max: float
    xfrc_applied_abs_max: float

    # 口径交叉验证
    jacobian_vs_index_max_diff: float
    accounting: Dict[str, Any]

    # 终末时刻的力分解（接触/限位/equality 拆开 + 肌肉主动/被动拆开）
    final_force_terms: Dict[str, Any]
    final_constraint_decomposition: Dict[str, Any]
    final_muscle_split: Dict[str, Any]

    # 终止附近逐步记录（含终止步）
    ledger_tail: List[Dict[str, Any]]

    trail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluatorConfig:
    """评估器配置。"""

    checkpoint_dir: Path = field(default_factory=lambda: paths.checkpoint_dir("LocomotionFull"))
    deterministic: bool = True
    device: str = "cpu"
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    research: Optional[ResearchEnvConfig] = None
    #: 规定的评估时长（秒）；仅研究模式生效，None 表示沿用官方 3.51 s
    max_episode_seconds: Optional[float] = None
    save_trajectory: bool = False
    save_ledger: bool = False
    #: 是否在终末时刻做约束力分解（稍贵，默认开）
    final_force_decomposition: bool = True
    render_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checkpoint_dir"] = str(self.checkpoint_dir)
        d["termination"] = self.termination.to_dict()
        d["research"] = self.research.to_dict() if self.research is not None else None
        return d

    def effective_research(self) -> ResearchEnvConfig:
        if self.research is not None:
            return self.research
        return ResearchEnvConfig(
            termination=self.termination,
            max_episode_seconds=self.max_episode_seconds,
        )


class LocomotionEvaluator:
    """封装「研究环境 + 官方策略 + 肌力缩放器」的评估器。

    环境一律通过 :func:`hemirl.research_env.build_research_env` 构建，
    以保证训练与评估共用同一套终止 / 计时 / 奖励语义。
    """

    def __init__(self, cfg: EvaluatorConfig):
        from hemirl import policy as policy_mod

        self.cfg = cfg
        self.env_cfg = OfficialEnvConfig.from_checkpoint(cfg.checkpoint_dir)
        self.research_cfg = cfg.effective_research()
        self.env, self.raw_env = build_research_env(
            env_config=self.env_cfg,
            env_cfg=self.research_cfg,
            render_mode=cfg.render_mode,
        )

        # self.raw_env 是 unwrapped 后的上游 LocomotionFullEnvV1
        self.model = self.raw_env.model
        self.data = self.raw_env.data
        self.mapping = muscle_groups.build_map(self.model)
        self.scaler = StrengthScaler(self.model, self.mapping)
        self.stack = policy_mod.load_sb3_stack(
            cfg.checkpoint_dir, self.env, device=cfg.device, deterministic=cfg.deterministic
        )
        self.policy_module = policy_mod

        self.pelvis_id = int(self.env.pelvis_id)
        self.dt = float(self.env.dt)
        self.frame_skip = int(self.env.frame_skip)
        self.nu = int(self.model.nu)
        self.act_indices: Dict[str, List[int]] = self._group_indices()

        # 实际加载的模型文件（用于 provenance 与「上游是否被改动」的核对）
        self.loaded_model_path = self._loaded_model_path()

    def _loaded_model_path(self) -> Optional[Path]:
        try:
            from hemirl import paths as _p

            return _p.MSGYM_ROOT / "msgym" / "MS-Human-700" / "MS-Human-700.xml"
        except Exception:
            return None

    def _group_indices(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for key in GROUP_KEYS:
            side, limb = key.split("/")
            out[key] = self.mapping.indices_for(side, limb)
        return out

    # ------------------------------------------------------------ 辅助

    @property
    def max_steps(self) -> int:
        return int(self.env.max_control_steps)

    def reference_speed(self) -> float:
        return float(self.raw_env.trajectory.get_trajectory_properties(0)[1])

    # ------------------------------------------------------------ episode

    def run_episode(
        self,
        seed: int,
        spec: Optional[StrengthSpec] = None,
        label: str = "",
        termination: Optional[TerminationConfig] = None,
        save_trajectory: Optional[bool] = None,
        trail_dir: Optional[Path] = None,
    ) -> EpisodeResult:
        """跑一个 episode，返回指标；可选保存逐步轨迹与 ledger。"""
        save_traj = self.cfg.save_trajectory if save_trajectory is None else save_trajectory
        if termination is not None:
            # 由环境重算派生量（time_limit_s / max_control_steps），避免两者不一致
            self.env.configure_termination(termination)

        # 每次 episode 都从基准重新设置肌力（禁止累乘）
        self.scaler.reset()
        applied: Optional[AppliedStrength] = None
        if spec is not None:
            if spec.mode != self.scaler.mode:
                self.scaler.mode = spec.mode
            applied = self.scaler.apply(spec)

        obs, _info = self.env.reset(seed=seed)
        ref_speed = self.reference_speed()
        root_start = root_state(self.model, self.data, self.pelvis_id)

        trail: Dict[str, List[np.ndarray]] = {
            "root_pos": [],
            "root_rot": [],
            "root_lin_vel": [],
            "root_ang_vel": [],
            "qpos": [],
            "qvel": [],
            "qacc": [],
            "action": [],
            "excitation": [],
            "activation": [],
            "actuator_force": [],
            "qpos_ref": [],
            "ncon": [],
            "total_normal_force": [],
            "reward": [],
        }

        group_acc: Dict[str, Dict[str, List[float]]] = {
            k: {"action_abs": [], "excitation": [], "activation": [], "force_abs": []} for k in GROUP_KEYS
        }
        qpos_track_err: List[float] = []
        pelvis_z: List[float] = []
        tilt: List[float] = []
        root_speed: List[float] = []
        root_ang_speed: List[float] = []
        ncon_hist: List[float] = []
        normal_force_hist: List[float] = []
        foot_contact_hist: List[float] = []
        action_abs: List[float] = []
        excitation_all: List[float] = []
        activation_all: List[float] = []
        force_abs_all: List[float] = []
        reward_totals = {"official_total": 0.0, "imitation": 0.0, "energy": 0.0,
                         "survival_physical": 0.0, "official_healthy": 0.0}
        max_abs_qvel = 0.0
        max_abs_qacc = 0.0
        qfrc_applied_max = 0.0
        xfrc_applied_max = 0.0
        jac_vs_index_max = 0.0
        steps_after_official_term = 0
        terminated = truncated = False
        reason: Optional[str] = None
        source: Optional[str] = None

        t0 = time.perf_counter()
        while True:
            normalized = self.stack.normalize_obs(obs)
            action = self.stack.predict(normalized)
            excitation = self.env.action(np.asarray(action))  # MuscleNormWrapper

            obs, reward, terminated, truncated, info = self.env.step(action)

            # ---- 以下所有记录都在 step **之后**，因此触发终止的最后一步同样被记录
            data = self.data
            rs = root_state(self.model, data, self.pelvis_id)
            jac_vs_index_max = max(
                jac_vs_index_max,
                float(np.max(np.abs(rs["lin_vel"] - rs["lin_vel_index_formula"]))),
                float(np.max(np.abs(rs["ang_vel"] - rs["ang_vel_index_formula"]))),
            )

            cm = contact_metrics(self.model, data)
            ncon_hist.append(cm["ncon"])
            normal_force_hist.append(cm["total_normal_force"])
            foot_contact_hist.append(cm["any_foot_contact"])

            act_arr = np.asarray(data.act, dtype=float)
            ctrl_arr = np.asarray(data.ctrl, dtype=float)
            force_arr = np.asarray(data.actuator_force, dtype=float)
            action_arr = np.asarray(action, dtype=float)

            for key in GROUP_KEYS:
                idx = self.act_indices[key]
                if not idx:
                    continue
                group_acc[key]["action_abs"].append(float(np.mean(np.abs(action_arr[idx]))))
                group_acc[key]["excitation"].append(float(np.mean(ctrl_arr[idx])))
                group_acc[key]["activation"].append(float(np.mean(act_arr[idx])))
                group_acc[key]["force_abs"].append(float(np.mean(np.abs(force_arr[idx]))))

            action_abs.append(float(np.mean(np.abs(action_arr))))
            excitation_all.append(float(np.mean(ctrl_arr)))
            activation_all.append(float(np.mean(act_arr)))
            force_abs_all.append(float(np.mean(np.abs(force_arr))))

            qpos_track_err.append(float(self.env.reference_track_err()))
            pelvis_z.append(float(rs["pos"][2]))
            tilt.append(float(self.env.up_tilt_deg()))
            root_speed.append(float(np.linalg.norm(rs["lin_vel"])))
            root_ang_speed.append(float(np.linalg.norm(rs["ang_vel"])))
            max_abs_qvel = max(max_abs_qvel, float(np.max(np.abs(data.qvel))))
            max_abs_qacc = max(max_abs_qacc, float(np.max(np.abs(data.qacc))))
            qfrc_applied_max = max(qfrc_applied_max, float(np.max(np.abs(data.qfrc_applied))))
            xfrc_applied_max = max(xfrc_applied_max, float(np.max(np.abs(data.xfrc_applied))))

            rc = info.get("reward_components") or {}
            reward_totals["official_total"] += float(reward) if self.research_cfg.reward_mode == REWARD_MODE_OFFICIAL else 0.0
            for k in ("imitation", "energy", "survival_physical", "official_healthy"):
                v = rc.get(k)
                if v is not None and np.isfinite(v):
                    reward_totals[k] += float(v)

            if info["termination"]["n_official_terminated_flags"] > 0 and not (terminated or truncated):
                steps_after_official_term += 1

            if save_traj:
                trail["root_pos"].append(rs["pos"])
                trail["root_rot"].append(rs["rot"].reshape(-1))
                trail["root_lin_vel"].append(rs["lin_vel"])
                trail["root_ang_vel"].append(rs["ang_vel"])
                trail["qpos"].append(np.asarray(data.qpos).copy())
                trail["qvel"].append(np.asarray(data.qvel).copy())
                trail["qacc"].append(np.asarray(data.qacc).copy())
                trail["action"].append(np.asarray(action, dtype=np.float32).copy())
                trail["excitation"].append(np.asarray(excitation, dtype=np.float32).copy())
                trail["activation"].append(np.asarray(act_arr, dtype=np.float32))
                trail["actuator_force"].append(np.asarray(force_arr, dtype=np.float32))
                trail["qpos_ref"].append(np.asarray(self.raw_env.qpos_ref, dtype=np.float32).copy())
                trail["ncon"].append(np.array([cm["ncon"]]))
                trail["total_normal_force"].append(np.array([cm["total_normal_force"]]))
                trail["reward"].append(np.array([float(reward)]))

            if terminated or truncated:
                reason = info["termination"]["termination_reason"]
                source = info["termination"]["termination_source"]
                break
        wall = time.perf_counter() - t0

        # 运行时间：实际仿真时间差（不是步数 × dt）
        meta = self.env.termination_metadata()
        sim_time = float(meta["elapsed_time_s"])
        n_steps = int(meta["n_control_steps"])
        root_end = root_state(self.model, self.data, self.pelvis_id)
        disp = np.asarray(root_end["pos"], dtype=float) - np.asarray(root_start["pos"], dtype=float)

        final_force_terms: Dict[str, Any] = {}
        final_constraint: Dict[str, Any] = {}
        final_muscle: Dict[str, Any] = {}
        if self.cfg.final_force_decomposition:
            final_force_terms = generalized_force_terms(self.model, self.data)
            final_constraint = decompose_constraint_forces(self.model, self.data)
            final_muscle = muscle_active_passive_split(self.model, self.data)

        ledger_tail: List[Dict[str, Any]] = []
        if self.env.ledger:
            for rec in self.env.ledger[-3:]:
                d = rec.to_dict()
                d.pop("termination_reason", None)  # 尾部已在 termination 字段中
                ledger_tail.append(d)

        result = EpisodeResult(
            label=label,
            seed=seed,
            strength=applied.summary() if applied is not None else {"spec": None},
            steps=n_steps,
            n_physics_steps=int(meta["n_physics_steps"]),
            control_dt_s=float(self.dt),
            sim_time_s=sim_time,
            sim_time_nominal_s=float(n_steps * self.dt),
            wall_time_s=float(wall),
            terminated=bool(meta["terminated"]),
            truncated=bool(meta["truncated"]),
            termination_reason=reason,
            termination_source=source,
            is_numeric_anomaly=bool(meta["numeric_anomaly"]),
            is_physical_fall=bool(meta["physical_fall"]),
            n_official_term_flags=int(meta["n_official_terminated_flags"]),
            n_steps_after_official_term=int(steps_after_official_term),
            reached_time_limit=bool(meta["reached_time_limit"] or meta["reached_official_time_limit"]),
            reached_step_cap=bool(meta["reached_step_cap"]),
            root_start_pos=[float(v) for v in root_start["pos"]],
            root_end_pos=[float(v) for v in root_end["pos"]],
            displacement_xy=float(np.linalg.norm(disp[:2])),
            forward_displacement_x=float(disp[0]),
            lateral_drift_y=float(disp[1]),
            distance_3d=float(np.linalg.norm(disp)),
            mean_speed_xy=float(np.linalg.norm(disp[:2]) / sim_time) if sim_time > 0 else 0.0,
            forward_speed_x=float(disp[0] / sim_time) if sim_time > 0 else 0.0,
            ref_forward_speed=ref_speed,
            mean_pelvis_height=float(np.mean(pelvis_z)) if pelvis_z else float("nan"),
            min_pelvis_height=float(np.min(pelvis_z)) if pelvis_z else float("nan"),
            final_pelvis_height=float(pelvis_z[-1]) if pelvis_z else float("nan"),
            max_pelvis_up_tilt_deg=float(np.max(tilt)) if tilt else float("nan"),
            final_pelvis_up_tilt_deg=float(tilt[-1]) if tilt else float("nan"),
            posture_change_deg=float(abs(tilt[-1] - tilt[0])) if len(tilt) >= 2 else 0.0,
            max_root_speed=float(np.max(root_speed)) if root_speed else float("nan"),
            max_root_ang_speed=float(np.max(root_ang_speed)) if root_ang_speed else float("nan"),
            final_root_lin_vel=[float(v) for v in root_end["lin_vel"]],
            final_root_ang_vel=[float(v) for v in root_end["ang_vel"]],
            mean_qpos_track_err=float(np.mean(qpos_track_err)) if qpos_track_err else float("nan"),
            max_qpos_track_err=float(np.max(qpos_track_err)) if qpos_track_err else float("nan"),
            final_qpos_track_err=float(qpos_track_err[-1]) if qpos_track_err else float("nan"),
            group_stats={
                k: {
                    "n_muscles": len(self.act_indices[k]),
                    "action_abs_mean": float(np.mean(v["action_abs"])) if v["action_abs"] else float("nan"),
                    "excitation_mean": float(np.mean(v["excitation"])) if v["excitation"] else float("nan"),
                    "activation_mean": float(np.mean(v["activation"])) if v["activation"] else float("nan"),
                    "force_abs_mean": float(np.mean(v["force_abs"])) if v["force_abs"] else float("nan"),
                }
                for k, v in group_acc.items()
            },
            action_abs_mean=float(np.mean(action_abs)) if action_abs else float("nan"),
            action_abs_max=float(np.max(action_abs)) if action_abs else float("nan"),
            excitation_mean=float(np.mean(excitation_all)) if excitation_all else float("nan"),
            activation_mean=float(np.mean(activation_all)) if activation_all else float("nan"),
            actuator_force_abs_mean=float(np.mean(force_abs_all)) if force_abs_all else float("nan"),
            reward_totals={k: float(v) for k, v in reward_totals.items()},
            ncon_mean=float(np.mean(ncon_hist)) if ncon_hist else float("nan"),
            ncon_max=float(np.max(ncon_hist)) if ncon_hist else float("nan"),
            total_normal_force_mean=float(np.mean(normal_force_hist)) if normal_force_hist else float("nan"),
            foot_contact_fraction=float(np.mean(foot_contact_hist)) if foot_contact_hist else float("nan"),
            max_abs_qvel=max_abs_qvel,
            max_abs_qacc=max_abs_qacc,
            qfrc_applied_abs_max=qfrc_applied_max,
            xfrc_applied_abs_max=xfrc_applied_max,
            jacobian_vs_index_max_diff=float(jac_vs_index_max),
            accounting=self.env.accounting_report(),
            final_force_terms=final_force_terms,
            final_constraint_decomposition=final_constraint,
            final_muscle_split=final_muscle,
            ledger_tail=ledger_tail,
        )

        if save_traj:
            out_dir = Path(trail_dir) if trail_dir else paths.RUNS_ROOT / "trajectories"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{label or 'episode'}_seed{seed}.npz"
            payload = {k: np.asarray(v, dtype=np.float32) for k, v in trail.items() if v}
            payload["sim_dt"] = np.array([self.dt], dtype=np.float64)
            payload["reset_time"] = np.array([self.env._reset_time], dtype=np.float64)
            np.savez_compressed(out_dir / name, **payload)
            result.trail = str(out_dir / name)

        if self.cfg.save_ledger:
            out_dir = (Path(trail_dir) if trail_dir else paths.RUNS_ROOT / "trajectories").parent
            out_dir.mkdir(parents=True, exist_ok=True)
            provenance.write_json(out_dir / f"ledger_{label or 'episode'}_seed{seed}.json", {
                "seed": seed,
                "label": label,
                "steps": self.env.ledger_to_dict(),
                "termination": meta,
            })

        self.scaler.reset()
        return result

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def __enter__(self) -> "LocomotionEvaluator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = [
    "EvaluatorConfig",
    "EpisodeResult",
    "LocomotionEvaluator",
    "contact_metrics",
    "root_state",
    "GROUP_KEYS",
]
