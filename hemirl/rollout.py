"""评估运行器：固定策略权重下的动力学 rollout，记录完整证据链。

一次 episode 的数据流（全部落盘可查）::

    策略输出 a_policy ∈ [-1,1]^700                （hemirl.policy）
      → MuscleNormWrapper: a_env = 1/(1+exp(-5(a-0.5)))  ∈ (0,1)^700
      → env.step(a_policy) 内部: data.ctrl[:] = a_env （gymnasium MujocoEnv）
      → MuJoCo muscle activation 动力学: data.act
      → 肌肉力: data.actuator_force = gain(len,vel)·act + bias(len)
      → 广义肌肉力矩: data.qfrc_actuator
      → 接触/约束/被动/重力: mj_step 求解
      → 下一时刻状态 data.qpos/qvel/qacc
      → 观测（含 qacc / actuator_force / act）→ 下一步策略输入

本模块同时记录 ``qfrc_applied``（人为外力/力矩，应为 0）与 ``xfrc_applied``（外力，应为 0），
以及接触与 equality 约束的力，用于区分「正常接触力 / 模型内部耦合约束 / 人为轨迹约束」。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hemirl import muscle_groups, paths, provenance
from hemirl.envs import OfficialEnvConfig, build_env
from hemirl.muscle_actuator import AppliedStrength, StrengthScaler, StrengthSpec
from hemirl.termination import (
    TerminationConfig,
    evaluate_research_termination,
    pelvis_upright_local_axis,
    upright_tilt_deg,
)

#: 分组键固定顺序，方便落盘对比
GROUP_KEYS = ("L/upper", "R/upper", "L/lower", "R/lower", "M/torso", "L/torso", "R/torso")


def contact_metrics(model, data) -> Dict[str, float]:
    """提取可稳定获得的接触指标。

    * ``ncon``：当前接触点数。
    * 法向力：用 ``mj_contactForce`` 读取每个接触的 6 维力（前 3 维为接触坐标系下的力，
      第 0 个分量是法向力），按 body 汇总。
    * 足部接触：脚（calcn/toes）是否与地面接触。
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
        for gid, sign in ((con.geom1, 1.0), (con.geom2, 1.0)):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
            if bname in foot_force:
                foot_force[bname] += sign * f_normal
    out["total_normal_force"] = total
    for n in foot_names:
        out[f"normal_force_{n}"] = foot_force[n]
    out["any_foot_contact"] = float(
        any(foot_force[n] > 1e-6 for n in foot_names)
    )
    return out


def root_state(model, data) -> Dict[str, np.ndarray]:
    """根节点状态：位置 / 姿态 / 速度（世界系）。

    **根节点 qpos/qvel 布局（实证，见 ``scripts/probe_root_layout.py``）**：
    ``pelvis`` body 的坐标系相对世界系绕 x 轴旋转了 −90°，因此关节名与实际世界方向
    并不一致：

    ==========  ================  ================================
    qpos 槽位   关节名             实际世界效果
    ==========  ================  ================================
    ``[0]``     ``pelvis_tz``     世界 **−y**（侧向平移）
    ``[1]``     ``pelvis_ty``     世界 **+z**（竖直平移）
    ``[2]``     ``pelvis_tx``     世界 **+x**（前进平移）
    ``[3]``     ``pelvis_tilt``   绕世界 y（俯仰，前倾/后仰）
    ``[4]``     ``pelvis_list``   绕世界 x（侧倾）
    ``[5]``     ``pelvis_rotation`` 绕世界 z（偏航）
    ==========  ================  ================================

    因此世界系线速度 = ``[qvel[2], -qvel[0], qvel[1]]``（由 `mj_jacBody` 实测的
    平移雅可比列 ``[1,0,0] / [0,-1,0] / [0,0,1]`` 推出）。
    """
    import mujoco

    pelvis = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    xmat = np.asarray(data.xmat[pelvis]).reshape(3, 3).copy()
    qvel = np.asarray(data.qvel)
    return {
        "pos": np.asarray(data.xpos[pelvis]).copy(),
        "rot": xmat,
        "lin_vel": np.array([qvel[2], -qvel[0], qvel[1]]).copy(),
        "ang_vel": np.array([qvel[4], -qvel[3], qvel[5]]).copy(),
    }


def up_tilt_deg(rot: np.ndarray, upright_local: Optional[np.ndarray] = None) -> float:
    """骨盆直立偏差角（度）。默认使用按参考姿态标定的直立轴，见
    :func:`hemirl.termination.pelvis_upright_local_axis`。"""
    return upright_tilt_deg(rot, upright_local)


@dataclass
class EpisodeResult:
    """一次 episode 的汇总指标。"""

    label: str
    seed: int
    strength: Dict[str, Any]
    steps: int
    sim_time_s: float
    wall_time_s: float
    terminated: bool
    truncated: bool
    termination_reason: Optional[str]
    termination_source: str
    n_official_term_flags: int
    n_steps_after_official_term: int

    # 根节点与整体运动
    root_start_pos: List[float]
    root_end_pos: List[float]
    displacement_xy: float
    distance_3d: float
    mean_speed_xy: float
    ref_forward_speed: float
    mean_pelvis_height: float
    min_pelvis_height: float
    max_pelvis_up_tilt_deg: float
    max_root_speed: float
    final_root_lin_vel: List[float]
    final_root_ang_vel: List[float]

    # 参考跟踪（仅作描述性指标，不作为终止依据）
    mean_qpos_track_err: float
    max_qpos_track_err: float

    # 肌肉 / 动作（分组统计）
    group_stats: Dict[str, Dict[str, float]]
    action_abs_mean: float
    action_abs_max: float
    excitation_mean: float
    activation_mean: float
    actuator_force_abs_mean: float

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
    save_trajectory: bool = False
    render_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checkpoint_dir"] = str(self.checkpoint_dir)
        d["termination"] = self.termination.to_dict()
        return d


class LocomotionEvaluator:
    """封装「官方环境 + 官方策略 + 肌力缩放器」的评估器。"""

    def __init__(self, cfg: EvaluatorConfig):
        import mujoco  # noqa: F401

        from hemirl import policy as policy_mod

        self.cfg = cfg
        self.env_cfg = OfficialEnvConfig.from_checkpoint(cfg.checkpoint_dir)
        self.env, self.raw_env = build_env(self.env_cfg, render_mode=cfg.render_mode)
        self.model = self.raw_env.unwrapped.model
        self.data = self.raw_env.unwrapped.data

        self.mapping = muscle_groups.build_map(self.model)
        self.scaler = StrengthScaler(self.model, self.mapping)
        self.stack = policy_mod.load_sb3_stack(
            cfg.checkpoint_dir, self.env, device=cfg.device, deterministic=cfg.deterministic
        )
        self.policy_module = policy_mod

        import mujoco as _mj

        self.pelvis_id = int(_mj.mj_name2id(self.model, _mj.mjtObj.mjOBJ_BODY, "pelvis"))
        self.dt = float(self.raw_env.unwrapped.dt)
        self.frame_skip = int(self.raw_env.unwrapped.frame_skip)
        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.act_indices: Dict[str, List[int]] = self._group_indices()

    # ------------------------------------------------------------ 辅助

    def _group_indices(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for key in GROUP_KEYS:
            side, limb = key.split("/")
            out[key] = self.mapping.indices_for(side, limb)
        return out

    def max_steps(self, termination: TerminationConfig) -> int:
        """本 episode 允许的最大步数。"""
        if termination.kind == "official":
            cycles = self.env_cfg.single_env_kwargs.get("gait_cycles", 3)
            return int(round(self.raw_env.unwrapped.terminate_time * cycles / self.dt))
        limit = termination.time_limit_s
        if limit is None:
            cycles = self.env_cfg.single_env_kwargs.get("gait_cycles", 3)
            limit = float(self.raw_env.unwrapped.terminate_time) * cycles
        return int(round(limit / self.dt))

    def reference_speed(self) -> float:
        return float(self.raw_env.unwrapped.trajectory.get_trajectory_properties(0)[1])

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
        """跑一个 episode，返回指标；可选保存逐步轨迹。"""
        term_cfg = termination or self.cfg.termination
        save_traj = self.cfg.save_trajectory if save_trajectory is None else save_trajectory

        # 每次 episode 都从基准重新设置肌力（禁止累乘）
        self.scaler.reset()
        applied: Optional[AppliedStrength] = None
        if spec is not None and spec.mode != self.scaler.mode:
            self.scaler.mode = spec.mode
        if spec is not None:
            applied = self.scaler.apply(spec)

        obs, _info = self.env.reset(seed=seed)
        max_steps = self.max_steps(term_cfg)

        # 按本回合的参考姿态标定骨盆直立轴（该模型 pelvis body 局部坐标系
        # 与世界「上」不对齐，必须标定后才能用倾角判据）
        upright_local = pelvis_upright_local_axis(
            self.model, np.asarray(self.raw_env.unwrapped.qpos_ref), self.pelvis_id
        )

        init_time = float(getattr(self.raw_env.unwrapped, "init_time", 0.0))
        root_start = root_state(self.model, self.data)
        ref_speed = self.reference_speed()

        trail: Dict[str, List[np.ndarray]] = {
            "root_pos": [],
            "root_rot": [],
            "root_lin_vel": [],
            "root_ang_vel": [],
            "qpos": [],
            "qvel": [],
            "action": [],
            "excitation": [],
            "activation": [],
            "actuator_force": [],
            "qpos_ref": [],
            "ncon": [],
            "total_normal_force": [],
        }

        group_acc: Dict[str, Dict[str, List[float]]] = {
            k: {"action_abs": [], "excitation": [], "activation": [], "force_abs": []}
            for k in GROUP_KEYS
        }
        qpos_track_err: List[float] = []
        pelvis_z: List[float] = []
        tilt: List[float] = []
        root_speed: List[float] = []
        ncon_hist: List[float] = []
        normal_force_hist: List[float] = []
        foot_contact_hist: List[float] = []
        action_abs: List[float] = []
        excitation_all: List[float] = []
        activation_all: List[float] = []
        force_abs_all: List[float] = []
        max_abs_qvel = 0.0
        max_abs_qacc = 0.0
        qfrc_applied_max = 0.0
        xfrc_applied_max = 0.0
        n_official_term = 0
        steps_after_official_term = 0
        reason: Optional[str] = None
        source = term_cfg.kind
        terminated = False
        truncated = False

        t0 = time.perf_counter()
        for step in range(max_steps):
            normalized = self.stack.normalize_obs(obs)
            action = self.stack.predict(normalized)
            excitation = self.env.action(np.asarray(action))  # MuscleNormWrapper

            obs, reward, term_official, trunc_official, info = self.env.step(action)

            if bool(term_official):
                n_official_term += 1
                if term_cfg.kind == "official":
                    terminated = True
                    reason = "官方规则: is_healthy 为假（偏离参考姿态超阈值）"
                    source = "official"
                    break
                steps_after_official_term += 1

            data = self.data
            rs = root_state(self.model, data)

            # 记录
            if save_traj:
                trail["root_pos"].append(rs["pos"])
                trail["root_rot"].append(rs["rot"].reshape(-1))
                trail["root_lin_vel"].append(rs["lin_vel"])
                trail["root_ang_vel"].append(rs["ang_vel"])
                trail["qpos"].append(np.asarray(data.qpos).copy())
                trail["qvel"].append(np.asarray(data.qvel).copy())
                trail["action"].append(np.asarray(action, dtype=np.float32).copy())
                trail["excitation"].append(np.asarray(excitation, dtype=np.float32).copy())
                trail["activation"].append(np.asarray(data.act, dtype=np.float32).copy())
                trail["actuator_force"].append(np.asarray(data.actuator_force, dtype=np.float32).copy())
                trail["qpos_ref"].append(np.asarray(self.raw_env.unwrapped.qpos_ref, dtype=np.float32).copy())

            cm = contact_metrics(self.model, data)
            ncon_hist.append(cm["ncon"])
            normal_force_hist.append(cm["total_normal_force"])
            foot_contact_hist.append(cm["any_foot_contact"])
            if save_traj:
                trail["ncon"].append(np.array([cm["ncon"]]))
                trail["total_normal_force"].append(np.array([cm["total_normal_force"]]))

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

            qref = np.asarray(self.raw_env.unwrapped.qpos_ref)
            qpos_track_err.append(float(np.mean(np.abs(np.asarray(data.qpos)[3:] - qref[3:]))))
            pelvis_z.append(float(rs["pos"][2]))
            tilt.append(up_tilt_deg(rs["rot"], upright_local))
            root_speed.append(float(np.linalg.norm(rs["lin_vel"])))
            max_abs_qvel = max(max_abs_qvel, float(np.max(np.abs(data.qvel))))
            max_abs_qacc = max(max_abs_qacc, float(np.max(np.abs(data.qacc))))
            qfrc_applied_max = max(qfrc_applied_max, float(np.max(np.abs(data.qfrc_applied))))
            xfrc_applied_max = max(xfrc_applied_max, float(np.max(np.abs(data.xfrc_applied))))

            # 研究终止规则（不因偏离参考而终止）
            r = evaluate_research_termination(
                self.model,
                data,
                term_cfg,
                self.pelvis_id,
                float(data.time),
                upright_local=upright_local,
            )
            if r is not None:
                terminated = True
                reason = r
                source = "research"
                break
            if bool(trunc_official):
                truncated = True
                break
        wall = time.perf_counter() - t0

        steps = len(pelvis_z)
        sim_time = steps * self.dt
        root_end = root_state(self.model, self.data)
        disp = root_end["pos"][:2] - root_start["pos"][:2]

        result = EpisodeResult(
            label=label,
            seed=seed,
            strength=applied.summary() if applied is not None else {"spec": None},
            steps=steps,
            sim_time_s=float(sim_time),
            wall_time_s=float(wall),
            terminated=bool(terminated),
            truncated=bool(truncated),
            termination_reason=reason if terminated else ("官方时间上限（truncated）" if truncated else None),
            termination_source=source,
            n_official_term_flags=int(n_official_term),
            n_steps_after_official_term=int(steps_after_official_term),
            root_start_pos=[float(v) for v in root_start["pos"]],
            root_end_pos=[float(v) for v in root_end["pos"]],
            displacement_xy=float(np.linalg.norm(disp)),
            distance_3d=float(np.linalg.norm(root_end["pos"] - root_start["pos"])),
            mean_speed_xy=float(np.linalg.norm(disp) / sim_time) if sim_time > 0 else 0.0,
            ref_forward_speed=ref_speed,
            mean_pelvis_height=float(np.mean(pelvis_z)) if pelvis_z else float("nan"),
            min_pelvis_height=float(np.min(pelvis_z)) if pelvis_z else float("nan"),
            max_pelvis_up_tilt_deg=float(np.max(tilt)) if tilt else float("nan"),
            max_root_speed=float(np.max(root_speed)) if root_speed else float("nan"),
            final_root_lin_vel=[float(v) for v in root_end["lin_vel"]],
            final_root_ang_vel=[float(v) for v in root_end["ang_vel"]],
            mean_qpos_track_err=float(np.mean(qpos_track_err)) if qpos_track_err else float("nan"),
            max_qpos_track_err=float(np.max(qpos_track_err)) if qpos_track_err else float("nan"),
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
            ncon_mean=float(np.mean(ncon_hist)) if ncon_hist else float("nan"),
            ncon_max=float(np.max(ncon_hist)) if ncon_hist else float("nan"),
            total_normal_force_mean=float(np.mean(normal_force_hist)) if normal_force_hist else float("nan"),
            foot_contact_fraction=float(np.mean(foot_contact_hist)) if foot_contact_hist else float("nan"),
            max_abs_qvel=max_abs_qvel,
            max_abs_qacc=max_abs_qacc,
            qfrc_applied_abs_max=qfrc_applied_max,
            xfrc_applied_abs_max=xfrc_applied_max,
        )

        if save_traj:
            out_dir = Path(trail_dir) if trail_dir else paths.RUNS_ROOT / "trajectories"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"{label or 'episode'}_seed{seed}.npz"
            payload = {k: np.asarray(v, dtype=np.float32) for k, v in trail.items() if v}
            payload["sim_dt"] = np.array([self.dt], dtype=np.float64)
            payload["init_time"] = np.array([init_time], dtype=np.float64)
            np.savez_compressed(out_dir / name, **payload)
            result.trail = str(out_dir / name)

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
    "up_tilt_deg",
    "GROUP_KEYS",
]
