"""正常行走微调用的附加奖励项（直立 / 前进 / 侧向 / 滑步）。

## 为什么需要附加项

上游奖励 = 参考轨迹模仿 + 能量 + ``w_healthy``。三项里**只有能量项不依赖参考**，
而 ``w_healthy`` 只在「偏离参考超阈值」时才归零。实测（20 s 长时诊断，
``reports/instability_diag.json``）：

* 跌倒前 ``imitation`` 已经从 ``-15.6``/步恶化到 ``-789``/步，而
  ``survival_physical`` 一直是 ``+100``/步（骨盆高度直到最后一刻都 > 0.55 m）。
  也就是说**只靠「物理存活」项无法在早期发现侧向失稳**；
* 侧向漂移 ``drift_y`` 从 ``0.02 m`` 涨到 ``0.44 m``，而这些信息在观测里被裁剪
  （``qpos`` 的侧向分量 ``z_raw`` 达 9.5–17.8，阈值 10）。

因此附加项只做三件事，且**每项都对应诊断里真实越界的信号**：

1. ``lateral_dev``：惩罚相对参考侧向路径的偏移（诊断中从 0.02 → 0.44 m）；
2. ``lateral_vel``：惩罚侧向速度（诊断中 ``com_vy`` 末段到 1.19 m/s）；
3. ``forward_shortfall``：前进速度不足时惩罚（诊断中 ``drift_x`` 末值 −0.63 m，
   人体落后参考）。这一项同时**防止「原地站立」退化**：站立时该项约
   ``w_forward_shortfall × v_target``，远大于正常行走时的值。
4. ``slip``：足部滑动超过容忍量时惩罚。

## 量级标定方法（先测量，再定权重）

不要凭感觉设权重。:func:`measure_nominal_magnitudes` 在**正常肌力 + 官方策略 +
短时协议**下跑一遍，测出每项的原始量（不加权），再由
:meth:`HealthyRewardConfig.calibrate` 把权重定成「每项均值 ≈ ``target_magnitude``」。
这样新增项与既有 ``imitation``（实测均值 ≈ −15.6/步）同量级而不会盖过它。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: 参考目标前进速度（来自官方轨迹元数据，m/s）
DEFAULT_FORWARD_SPEED_TARGET = 1.0378


@dataclass
class HealthyRewardConfig:
    """附加奖励项配置。

    Attributes:
        enable: 是否启用（False 时 ``compute`` 全返回 0，行为与上游一致）。
        w_lateral_dev: 侧向偏移惩罚权重（对 |Δy| 线性）。
        w_lateral_vel: 侧向速度惩罚权重（对 |Δvy| 线性）。
        w_forward_shortfall: 前进速度不足惩罚权重（对 max(0, v_target − v_x) 线性）。
        w_slip: 滑步惩罚权重（对 max(0, slip − slip_tolerance) 线性）。
        forward_speed_target: 目标前进速度（m/s）。
        slip_tolerance: 允许的足部滑动（m/s），低于此值不惩罚。
        reference_lateral_from: 相对哪个量计算侧向偏移。
            ``'ref'`` = 与参考的侧向位置比较（推荐，参考本身有 ±0.05 m 摆动）；
            ``'start'`` = 与回合起始侧向位置比较。
    """

    enable: bool = False
    w_lateral_dev: float = 0.0
    w_lateral_vel: float = 0.0
    w_forward_shortfall: float = 0.0
    w_slip: float = 0.0
    forward_speed_target: float = DEFAULT_FORWARD_SPEED_TARGET
    slip_tolerance: float = 1.0
    reference_lateral_from: str = "ref"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -------------------------------------------------------- 标定

    def calibrate(
        self,
        nominal: Dict[str, float],
        target_magnitude: float = 5.0,
    ) -> "HealthyRewardConfig":
        """按「正常行走时每项均值 ≈ ``target_magnitude``」反解权重。

        Args:
            nominal: :func:`measure_nominal_magnitudes` 的返回值（未加权原始量）。
            target_magnitude: 每项在正常行走时的目标惩罚量级（每步，
                与 ``imitation`` 的 −15.6/步 相比属同量级但更小）。
        """
        def w(key: str) -> float:
            v = float(nominal.get(key, 0.0))
            return float(target_magnitude / v) if v > 1e-9 else 0.0

        self.enable = True
        self.w_lateral_dev = w("lateral_dev_abs_mean")
        self.w_lateral_vel = w("lateral_vel_abs_mean")
        self.w_forward_shortfall = w("forward_shortfall_mean")
        self.w_slip = w("slip_excess_mean")
        return self


def measure_nominal_magnitudes(
    env,
    evaluator,
    seed: int = 0,
    n_steps: int = 175,
) -> Dict[str, float]:
    """在正常肌力 + 官方策略下测量各附加项的**未加权原始量**。

    返回均值，供 :meth:`HealthyRewardConfig.calibrate` 反解权重。
    """
    obs, _ = env.reset(seed=seed)
    raw = evaluator.raw_env
    acc: Dict[str, list] = {}
    for _ in range(n_steps):
        norm = evaluator.stack.normalize_obs(obs)
        action = evaluator.stack.predict(norm)
        obs, _r, terminated, truncated, _info = env.step(action)
        m = raw_measure(evaluator, raw)
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
        if terminated or truncated:
            break
    return {
        "lateral_dev_abs_mean": float(np.mean(acc["lateral_dev_abs"])) if acc.get("lateral_dev_abs") else 0.0,
        "lateral_vel_abs_mean": float(np.mean(acc["lateral_vel_abs"])) if acc.get("lateral_vel_abs") else 0.0,
        "forward_shortfall_mean": float(np.mean(acc["forward_shortfall"])) if acc.get("forward_shortfall") else 0.0,
        "slip_excess_mean": float(np.mean(acc["slip_excess"])) if acc.get("slip_excess") else 0.0,
        "forward_speed_mean": float(np.mean(acc["forward_speed"])) if acc.get("forward_speed") else 0.0,
        "slip_mean": float(np.mean(acc["slip"])) if acc.get("slip") else 0.0,
        "n_steps": len(acc.get("lateral_dev_abs", [])),
    }


def raw_measure(evaluator, raw_env) -> Dict[str, float]:
    """取当前状态下的未加权原始量（供标定与诊断复用）。"""
    import mujoco

    data = evaluator.data
    model = evaluator.model
    pid = evaluator.pelvis_id
    from hemirl.rollout import root_state

    rs = root_state(model, data, pid)
    qref = np.asarray(raw_env.qpos_ref, dtype=float)
    # 参考的世界侧向位置：qpos[0] = pelvis_tz → 世界 −y
    ref_y = -float(qref[0])
    ref_vx = float(getattr(raw_env, "_last_ref_vel_x", 0.0))
    lateral_dev = abs(float(rs["pos"][1]) - ref_y)
    lateral_vel = abs(float(rs["lin_vel"][1]) - 0.0)
    forward_shortfall = max(0.0, DEFAULT_FORWARD_SPEED_TARGET - float(rs["lin_vel"][0]))
    # 足部滑动：接触且法向力 > 20 N 的足部 body 的水平速度
    slip = 0.0
    buf = np.zeros(6, dtype=np.float64)
    geom_f: Dict[int, float] = {}
    for i in range(int(data.ncon)):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, buf)
        f = float(abs(buf[0]))
        for gid in (int(con.geom1), int(con.geom2)):
            geom_f[gid] = max(geom_f.get(gid, 0.0), f)
    for name in ("calcn_r", "calcn_l", "toes_r", "toes_l"):
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        if bid < 0:
            continue
        adr, num = int(model.body_geomadr[bid]), int(model.body_geomnum[bid])
        if any(geom_f.get(adr + g, 0.0) > 20.0 for g in range(num)):
            slip = max(slip, float(np.linalg.norm(np.asarray(data.cvel[bid][3:6], dtype=float))))
    return {
        "lateral_dev_abs": lateral_dev,
        "lateral_vel_abs": lateral_vel,
        "forward_shortfall": forward_shortfall,
        "slip_excess": max(0.0, slip - 1.0),
        "lateral_dev_signed": float(rs["pos"][1]) - ref_y,
        "forward_speed": float(rs["lin_vel"][0]),
        "slip": slip,
    }


def compute(
    cfg: HealthyRewardConfig,
    *,
    pelvis_y: float,
    ref_y: float,
    pelvis_y_start: float,
    lin_vel: np.ndarray,
    slip: float,
) -> Tuple[Dict[str, float], float]:
    """计算附加项。返回 ``(各项分量, 合计)``。

    正值表示惩罚被加到总奖励上（本函数返回的是**负的**惩罚，便于直接相加）。
    """
    if not cfg.enable:
        zero = {
            "extra_lateral_dev": 0.0,
            "extra_lateral_vel": 0.0,
            "extra_forward_shortfall": 0.0,
            "extra_slip": 0.0,
        }
        return zero, 0.0

    ref = float(ref_y) if cfg.reference_lateral_from == "ref" else float(pelvis_y_start)
    terms = {
        "extra_lateral_dev": -cfg.w_lateral_dev * abs(float(pelvis_y) - ref),
        "extra_lateral_vel": -cfg.w_lateral_vel * abs(float(lin_vel[1])),
        "extra_forward_shortfall": -cfg.w_forward_shortfall
        * max(0.0, cfg.forward_speed_target - float(lin_vel[0])),
        "extra_slip": -cfg.w_slip * max(0.0, float(slip) - cfg.slip_tolerance),
    }
    return terms, float(sum(terms.values()))


__all__ = [
    "HealthyRewardConfig",
    "measure_nominal_magnitudes",
    "raw_measure",
    "compute",
    "DEFAULT_FORWARD_SPEED_TARGET",
]
