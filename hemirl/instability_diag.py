"""长时行走失稳诊断：把逐步信号全部落盘，用「谁先越界」定位失稳起因。

## 为什么要按组统计观测

观测 3601 维是**拼接**出来的，不同分段的物理含义与量级完全不同。只看整体
均值/标准差会把「700 维肌肉激活」淹没「6 维足端位置」。因此本模块按上游
``_get_obs`` 的拼接顺序给出**精确切片**，并逐组统计：

* 原始值范围；
* 经官方 ``VecNormalize`` 标准化后的数值范围；
* **裁剪比例**（``|标准化| > clip_obs`` 的占比）——这是「策略看到的信息被削掉」的直接证据。

重点检查三段与根节点/参考有关的分量：``qpos[0:3]``（绝对根位置）、``qpos_ref``（当前参考）、
``qpos_ref_future``（未来参考）。

## 参考自身缺口 vs 人体实际漂移（必须分开）

上游参考是**单条周期性轨迹**，它自身在循环边界有非闭合缺口（见
:func:`hemirl.research_env.reference_continuity_report`）。因此要区分：

* ``ref_*`` 系列：**参考自己**的位置/速度/夹角，用来看参考是否在漂；
* ``drift_*`` 系列：**人体相对参考**的偏差，用来判断人体是否真的在积累误差。

**不要**把「每周期姿态缺口 × 周期数」直接当成人体的姿态误差——那是参考自身的属性。

## 领先指标分析

对每个信号，用**回合前 ``baseline_window_s`` 秒**建立基线（mean ± k·std），
再找该信号第一次越界的时间。把各信号的首次越界时间排序，就能看出**谁先动**，
而不是等到跌倒了再回头猜原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: 观测分组（按上游 ``LocomotionFullEnvV1._get_obs`` 的拼接顺序）
#: 总长 85 + 85 + 85 + 700 + 700 + 700 + 700 + 18 + 85 + 425 + 18 = 3601
OBS_GROUPS: Tuple[Tuple[str, int], ...] = (
    ("qpos", 85),
    ("qvel", 85),
    ("qacc", 85),
    ("act", 700),
    ("actuator_forces", 700),
    ("actuator_length", 700),
    ("actuator_velocity", 700),
    ("key_xpos", 18),
    ("qpos_ref", 85),
    ("qpos_ref_future", 425),
    ("key_xpos_ref", 18),
)

OBS_TOTAL = 3601

#: 需要重点关注的分量（与根节点 / 参考直接相关）
FOCUS_COMPONENTS = {
    "root_abs_pos": ("qpos", slice(0, 3)),
    "root_abs_vel": ("qvel", slice(0, 3)),
    "ref_cur_pos": ("qpos_ref", slice(0, 3)),
    "ref_cur_vel": ("qpos_ref_future", slice(0, 3)),  # 未来参考的第 0 个时间片即「下一步参考」
    "root_ref_err": ("qpos", slice(0, 3)),
}


def obs_group_slices() -> Dict[str, slice]:
    """返回每个观测分组的切片；总长必须等于 3601。"""
    out: Dict[str, slice] = {}
    start = 0
    for name, n in OBS_GROUPS:
        out[name] = slice(start, start + n)
        start += n
    if start != OBS_TOTAL:
        raise AssertionError(f"观测分组总长 {start} != {OBS_TOTAL}")
    return out


@dataclass
class DiagConfig:
    """诊断配置。"""

    dt: float = 0.02
    baseline_window_s: float = 1.0
    k_sigma: float = 6.0
    clip_obs: float = 10.0
    #: 低于此高度视为「已处于失稳末段」，用于分阶段汇总
    late_stage_height_m: float = 0.80
    #: 判定「足部处于支撑」所需的最小法向力（N）。低于该值视为轻擦，不计入滑动统计。
    slip_min_normal_force: float = 20.0


class InstabilityDiagnoser:
    """逐步采集 + 首次越界定位。"""

    def __init__(self, evaluator, cfg: Optional[DiagConfig] = None):
        import mujoco

        self.ev = evaluator
        self.model = evaluator.model
        self.data = evaluator.data
        self.cfg = cfg or DiagConfig()
        self.slices = obs_group_slices()
        self.mujoco = mujoco

        self.pelvis_id = int(evaluator.pelvis_id)
        self.joint_qposadr = {
            n: int(self.model.joint(n).qposadr[0])
            for n in ("pelvis_tx", "pelvis_ty", "pelvis_tz")
        }

        # VecNormalize 的统计量：裁剪发生在 normalize_obs 内部，
        # 因此要判断「是否被裁剪」必须自己算**裁剪前**的标准化值。
        vn = evaluator.stack.vec_normalize
        self.obs_mean = np.asarray(vn.obs_rms.mean, dtype=float)
        self.obs_var = np.asarray(vn.obs_rms.var, dtype=float)
        self.obs_eps = float(getattr(vn, "epsilon", 1e-8))
        self.clip_obs = float(getattr(vn, "clip_obs", 10.0))
        # 左右下肢组下标
        from hemirl import muscle_groups

        self.mapping = muscle_groups.build_map(self.model)
        self.lower_idx = {
            side: np.asarray(self.mapping.indices_for(side, "lower"), dtype=int)
            for side in ("L", "R")
        }
        self.upper_idx = {
            side: np.asarray(self.mapping.indices_for(side, "upper"), dtype=int)
            for side in ("L", "R")
        }
        self.foot_bodies: Dict[str, int] = {}
        for name in ("calcn_r", "calcn_l", "toes_r", "toes_l"):
            bid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))
            if bid >= 0:
                self.foot_bodies[name] = bid

    # ---------------------------------------------------------- 单步采集

    def _foot_state(self) -> Dict[str, Any]:
        """足部接触与滑动。

        * 接触判定：足部 body 的 geom 与地面的接触，且**法向力超过阈值**
          （``slip_min_normal_force``）。只按 geom 出现与否会把脚尖轻擦也算成支撑，
          实测方差很大（均值得 2.4 m/s 量级的假滑动）。
        * 滑动代理量：接触期间该 body 原点的水平速度（m/s）。这是可稳定获得的代理量，
          不等同于接触点切向速度，但足以分辨「踩住」与「滑步」。
        """
        mj = self.mujoco
        data = self.data
        model = self.model
        buf = np.zeros(6, dtype=np.float64)
        # geom -> 最大法向力
        geom_normal: Dict[int, float] = {}
        for i in range(int(data.ncon)):
            con = data.contact[i]
            mj.mj_contactForce(model, data, i, buf)
            f = float(abs(buf[0]))
            for gid in (int(con.geom1), int(con.geom2)):
                if f > geom_normal.get(gid, 0.0):
                    geom_normal[gid] = f
        contacted: Dict[str, bool] = {}
        for name, bid in self.foot_bodies.items():
            adr = int(model.body_geomadr[bid])
            num = int(model.body_geomnum[bid])
            contacted[name] = any(
                geom_normal.get(adr + g, 0.0) > self.cfg.slip_min_normal_force
                for g in range(num)
            )
        slip: Dict[str, float] = {}
        for name, bid in self.foot_bodies.items():
            v = np.asarray(data.cvel[bid][3:6], dtype=float)  # cvel 的 [3:6] 是世界系线速度
            slip[name] = float(np.linalg.norm(v)) if contacted.get(name) else 0.0
        left = bool(contacted.get("calcn_l") or contacted.get("toes_l"))
        right = bool(contacted.get("calcn_r") or contacted.get("toes_r"))
        return {
            "contact_left": left,
            "contact_right": right,
            "n_feet_in_contact": int(left) + int(right),
            "foot_slip_max": float(max(slip.values())) if slip else 0.0,
            "foot_slip": slip,
        }

    def _com_state(self, prev_com: Optional[np.ndarray], prev_t: Optional[float]) -> Dict[str, Any]:
        """全身质心位置与速度。

        ``data.subtree_com[0]`` 是整棵模型的质心（世界系）。质心速度用相邻控制步的
        差分得到，因此带一点离散误差，这里显式标注为
        ``com_vel_method='finite_difference_control_dt'``。
        """
        com = np.asarray(self.data.subtree_com[0], dtype=float).copy()
        t = float(self.data.time)
        if prev_com is None or prev_t is None or t <= prev_t:
            com_vel = np.zeros(3)
        else:
            com_vel = (com - prev_com) / (t - prev_t)
        return {
            "com": com,
            "com_vel": com_vel,
            "com_vel_method": "finite_difference_control_dt",
        }

    def _obs_stats(self, raw_obs: np.ndarray, norm_obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """按组统计**裁剪前**标准化值与真正的裁剪比例。

        注意：``VecNormalize.normalize_obs`` 会直接 ``np.clip`` 到 ±``clip_obs``，
        因此直接看它的输出永远得到 ``|z| <= clip_obs``，「裁剪比例」恒为 0。
        必须自己算 ``z_raw = (obs - mean) / sqrt(var + eps)`` 再与 ``clip_obs`` 比较，
        才能知道有多少维被削掉。
        """
        out: Dict[str, Dict[str, float]] = {}
        for name, sl in self.slices.items():
            r = np.asarray(raw_obs[sl], dtype=float)
            z_raw = (r - self.obs_mean[sl]) / np.sqrt(self.obs_var[sl] + self.obs_eps)
            n_clip = int(np.sum(np.abs(z_raw) > self.clip_obs))
            out[name] = {
                "raw_absmax": float(np.max(np.abs(r))) if r.size else 0.0,
                "raw_mean": float(np.mean(r)) if r.size else 0.0,
                "z_raw_absmax": float(np.max(np.abs(z_raw))) if z_raw.size else 0.0,
                "z_raw_std": float(np.std(z_raw)) if z_raw.size else 0.0,
                "n_clipped": float(n_clip),
                "clip_fraction": float(n_clip / z_raw.size) if z_raw.size else 0.0,
                "norm_absmax": float(np.max(np.abs(np.asarray(norm_obs[sl], dtype=float))))
                if z_raw.size
                else 0.0,
            }
        return out

    def focus_components(self, raw_obs: np.ndarray) -> Dict[str, Dict[str, float]]:
        """重点分量的逐分量诊断：绝对根位置 / 根速度 / 当前参考 / 未来参考。

        返回每个分量组的 z_raw 绝对值、是否被裁剪、以及相对训练统计的偏移倍数。
        """
        raw_obs = np.asarray(raw_obs, dtype=float)
        groups = {
            "root_abs_pos(qpos[0:3])": ("qpos", slice(0, 3)),
            "root_abs_rot(qpos[3:6])": ("qpos", slice(3, 6)),
            "root_abs_vel(qvel[0:3])": ("qvel", slice(0, 3)),
            "ref_cur_pos(qpos_ref[0:3])": ("qpos_ref", slice(0, 3)),
            "ref_next_pos(qpos_ref_future[0:3])": ("qpos_ref_future", slice(0, 3)),
        }
        out: Dict[str, Dict[str, float]] = {}
        for label, (gname, sub) in groups.items():
            sl = self.slices[gname]
            lo, hi = sl.start + sub.start, sl.start + sub.stop
            r = raw_obs[lo:hi]
            z = (r - self.obs_mean[lo:hi]) / np.sqrt(self.obs_var[lo:hi] + self.obs_eps)
            out[label] = {
                "n_clipped": float(np.sum(np.abs(z) > self.clip_obs)),
                "z_absmax": float(np.max(np.abs(z))) if z.size else 0.0,
                "z_abs": [float(v) for v in np.abs(z)],
                "raw": [float(v) for v in r],
            }
        return out

    # ---------------------------------------------------------- 整回合

    def run(self, seed: int, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """跑一个回合并返回逐步诊断序列与汇总。"""
        ev = self.ev
        env = ev.env
        raw = ev.raw_env
        obs, _info = env.reset(seed=seed)

        pose_ref = np.asarray(raw.qpos_ref, dtype=float).copy()
        n_max = max_steps if max_steps is not None else int(env.max_control_steps)
        series: Dict[str, List[float]] = {}
        obs_group_series: Dict[str, List[Dict[str, float]]] = {n: [] for n in self.slices}

        prev_com: Optional[np.ndarray] = None
        prev_t: Optional[float] = None
        terminated = truncated = False
        reason = source = None

        for step in range(n_max):
            norm_obs = np.asarray(ev.stack.normalize_obs(obs), dtype=np.float32)
            action = np.clip(np.asarray(ev.stack.predict(norm_obs), dtype=float), -1.0, 1.0)

            # 记录**动作执行前**的观测统计（即策略实际看到的东西）
            raw_obs = np.asarray(obs, dtype=float)
            gstats = self._obs_stats(raw_obs, norm_obs)
            for name in self.slices:
                obs_group_series[name].append(gstats[name])
            focus = self.focus_components(raw_obs)

            obs, reward, terminated, truncated, info = env.step(action)

            data = self.data
            rs_pos = np.asarray(data.xpos[self.pelvis_id], dtype=float)
            from hemirl.rollout import root_state

            rs = root_state(self.model, data, self.pelvis_id)
            com = self._com_state(prev_com, prev_t)
            prev_com, prev_t = com["com"], float(data.time)
            feet = self._foot_state()

            act = np.asarray(data.act, dtype=float)
            ctrl = np.asarray(data.ctrl, dtype=float)
            force = np.asarray(data.actuator_force, dtype=float)
            qref = np.asarray(raw.qpos_ref, dtype=float)
            rc = info.get("reward_components") or {}
            ped = info.get("termination") or {}

            rec: Dict[str, float] = {
                "t": float(data.time),
                "step": float(step),
                "pelvis_x": float(rs_pos[0]),
                "pelvis_y": float(rs_pos[1]),
                "pelvis_z": float(rs_pos[2]),
                "pelvis_vx": float(rs["lin_vel"][0]),
                "pelvis_vy": float(rs["lin_vel"][1]),
                "pelvis_vz": float(rs["lin_vel"][2]),
                "pelvis_speed_h": float(np.linalg.norm(rs["lin_vel"][:2])),
                "pelvis_wx": float(rs["ang_vel"][0]),
                "pelvis_wy": float(rs["ang_vel"][1]),
                "pelvis_wz": float(rs["ang_vel"][2]),
                "pelvis_w_norm": float(np.linalg.norm(rs["ang_vel"])),
                "up_tilt_deg": float(env.up_tilt_deg()),
                "com_x": float(com["com"][0]),
                "com_y": float(com["com"][1]),
                "com_z": float(com["com"][2]),
                "com_vx": float(com["com_vel"][0]),
                "com_vy": float(com["com_vel"][1]),
                "com_speed_h": float(np.linalg.norm(com["com_vel"][:2])),
                "contact_left": float(feet["contact_left"]),
                "contact_right": float(feet["contact_right"]),
                "n_feet_in_contact": float(feet["n_feet_in_contact"]),
                "double_support": float(feet["contact_left"] and feet["contact_right"]),
                "no_support": float(not feet["contact_left"] and not feet["contact_right"]),
                "foot_slip_max": float(feet["foot_slip_max"]),
                "act_mean": float(np.mean(act)),
                "act_L_lower": float(np.mean(act[self.lower_idx["L"]])),
                "act_R_lower": float(np.mean(act[self.lower_idx["R"]])),
                "act_L_upper": float(np.mean(act[self.upper_idx["L"]])),
                "act_R_upper": float(np.mean(act[self.upper_idx["R"]])),
                "act_asym_lower": float(
                    np.mean(act[self.lower_idx["R"]]) - np.mean(act[self.lower_idx["L"]])
                ),
                "force_abs_mean": float(np.mean(np.abs(force))),
                "ctrl_mean": float(np.mean(ctrl)),
                "action_absmean": float(np.mean(np.abs(action))),
                "action_sat_frac": float(np.mean(np.abs(action) > 0.99)),
                "action_sat_frac_L_lower": float(
                    np.mean(np.abs(action[self.lower_idx["L"]]) > 0.99)
                ),
                "action_sat_frac_R_lower": float(
                    np.mean(np.abs(action[self.lower_idx["R"]]) > 0.99)
                ),
                "reward_total": float(reward),
                "reward_imitation": float(rc.get("imitation", float("nan"))),
                "reward_energy": float(rc.get("energy", float("nan"))),
                "reward_survival_physical": float(rc.get("survival_physical", float("nan"))),
                "reward_official_healthy": float(rc.get("official_healthy", float("nan"))),
                "ref_pelvis_x": float(qref[2]),
                "ref_pelvis_y": float(-qref[0]),
                "ref_pelvis_z": float(qref[1]),
                "ref_y_minus_ref_y0": float(-qref[0] - (-pose_ref[0])),
                "drift_y_human_minus_ref": float(rs_pos[1] - (-qref[0])),
                "drift_x_human_minus_ref": float(rs_pos[0] - qref[2]),
                "qpos_track_err": float(env.reference_track_err()),
                "ref_cycle": float(data.time / float(raw.terminate_time)),
                "n_official_term_flags": float(ped.get("n_official_terminated_flags", 0)),
            }
            for k, v in rec.items():
                series.setdefault(k, []).append(v)
            for label, st in focus.items():
                series.setdefault(f"zraw_absmax[{label}]", []).append(float(st["z_absmax"]))
                series.setdefault(f"nclip[{label}]", []).append(float(st["n_clipped"]))

            if terminated or truncated:
                reason = ped.get("termination_reason")
                source = ped.get("termination_source")
                break

        return self.summarize(
            seed, series, obs_group_series, reason, source,
            terminated=terminated, truncated=truncated,
        )

    # ---------------------------------------------------------- 汇总

    def summarize(
        self,
        seed: int,
        series: Dict[str, List[float]],
        obs_group_series: Dict[str, List[Dict[str, float]]],
        reason: Optional[str],
        source: Optional[str],
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, Any]:
        """按「前 baseline 窗口」建立基线，找每个信号首次越界的时间。"""
        t = np.asarray(series["t"], dtype=float)
        n = t.size
        base_n = min(max(3, int(self.cfg.baseline_window_s / self.cfg.dt)), max(3, n // 4))

        lead: Dict[str, Any] = {}
        skip = {"t", "step", "n_official_term_flags", "ref_cycle"} if False else {
            "t", "step", "n_official_term_flags", "ref_cycle"
        }
        for name, vals in series.items():
            a = np.asarray(vals, dtype=float)
            if a.size != n or name in skip or not np.all(np.isfinite(a)):
                continue
            b = a[:base_n]
            mu, sd = float(np.mean(b)), float(np.std(b))
            if sd < 1e-12:
                sd = max(1e-9, abs(mu) * 1e-3)
            thr_hi = mu + self.cfg.k_sigma * sd
            thr_lo = mu - self.cfg.k_sigma * sd
            outside = (a > thr_hi) | (a < thr_lo)
            first = int(np.argmax(outside)) if outside.any() else -1
            lead[name] = {
                "baseline_mean": mu,
                "baseline_std": sd,
                "final": float(a[-1]),
                "absmax": float(np.max(np.abs(a))),
                "first_outside_index": first,
                "first_outside_t": float(t[first]) if first >= 0 else None,
                "thr_hi": thr_hi,
                "thr_lo": thr_lo,
            }

        ordered = sorted(
            (
                (float(v["first_outside_t"]), k)
                for k, v in lead.items()
                if v["first_outside_t"] is not None
            ),
            key=lambda kv: kv[0],
        )

        obs_summary: Dict[str, Any] = {}
        for name, rows in obs_group_series.items():
            if not rows:
                continue
            clip = np.asarray([r["clip_fraction"] for r in rows], dtype=float)
            nclip = np.asarray([r["n_clipped"] for r in rows], dtype=float)
            zmax = np.asarray([r["z_raw_absmax"] for r in rows], dtype=float)
            first_clip = int(np.argmax(nclip > 0)) if (nclip > 0).any() else -1
            obs_summary[name] = {
                "clip_fraction_mean": float(np.mean(clip)),
                "clip_fraction_max": float(np.max(clip)),
                "n_clipped_mean": float(np.mean(nclip)),
                "n_clipped_max": float(np.max(nclip)),
                "dim": int(rows[0].get("n_clipped", 0) / max(rows[0]["clip_fraction"], 1e-12))
                if rows[0]["clip_fraction"] > 0
                else 0,
                "first_clip_t": float(t[first_clip]) if first_clip >= 0 else None,
                "z_raw_absmax_max": float(np.max(zmax)),
                "raw_absmax_max": float(np.max([r["raw_absmax"] for r in rows])),
            }

        return {
            "seed": seed,
            "n_steps": n,
            "alive_time_s": float(t[-1]) if n else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_source": source,
            "termination_reason": reason,
            "obs_group_summary": obs_summary,
            "leading_indicators": ordered[:20],
            "signal_stats": lead,
            "series": series,
            "config": {
                "dt": self.cfg.dt,
                "baseline_window_s": self.cfg.baseline_window_s,
                "k_sigma": self.cfg.k_sigma,
                "clip_obs": self.cfg.clip_obs,
                "baseline_n_steps": base_n,
            },
        }


__all__ = [
    "DiagConfig",
    "InstabilityDiagnoser",
    "OBS_GROUPS",
    "OBS_TOTAL",
    "obs_group_slices",
]
