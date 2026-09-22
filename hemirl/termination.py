"""终止条件：官方复现规则 + 独立研究规则。

## 官方规则（用于复现，原样保留）

``msgym/envs/locomotionFull_v1.py``::

    is_healthy   = mean(|qpos[3:] - qpos_ref[3:]|) <= qpos_diff_th   # 官方默认 0.06
    terminated   = not is_healthy or time >= terminate_time * cycles
    truncated    = time >= terminate_time * cycles

即**偏离参考姿态**就终止，与任务书要求的研究条件不同。

## 研究规则（本模块新增）

不再因「偏离参考姿态」而终止，改用物理上明确的跌倒/数值异常/超时条件：

===================  ==========================================  ==========================
条件                  坐标含义与判据                                 参数来源
===================  ==========================================  ==========================
低谷盆高度            ``data.xpos[pelvis_id][2]``（世界系 z，米），   默认 0.55 m ≈ 参考轨迹
                     z 轴向上；低于阈值判为跌倒                     骨盆高度均值 0.9205 m 的 60%
骨盆直立偏差          以参考姿态标定的骨盆直立轴与世界 z 轴的夹角     默认 60°（直立为 0°），见
                     （见 :func:`pelvis_upright_local_axis`）        下方说明为何需要标定
根节点速度            ``data.qvel[0:3]`` 是 pelvis_tz/ty/tx 三个      默认 10 m/s；参考步速约 1.04 m/s，
                     滑动关节的速度（范数与坐标顺序无关）            10 倍余量只捕捉数值爆炸
根节点角速度          ``data.qvel[3:6]`` 是 pelvis_tilt/list/         默认 50 rad/s
                     rotation 三个铰链的角速度
数值异常              qpos/qvel/qacc/act 出现 NaN 或 Inf             模型与积分器（Euler, dt=2 ms）
超时                  仿真时间上限                                   默认取官方 ``terminate_time*cycles``
===================  ==========================================  ==========================

**根节点 qpos 布局**（实证，见 ``scripts/probe_root_layout.py``）::

    qpos[0] = pelvis_tz -> 世界 -y（侧向）
    qpos[1] = pelvis_ty -> 世界 +z（**竖直**）
    qpos[2] = pelvis_tx -> 世界 +x（前进）
    qpos[3] = pelvis_tilt -> 绕世界 y（俯仰）
    qpos[4] = pelvis_list -> 绕世界 x（侧倾）
    qpos[5] = pelvis_rotation -> 绕世界 z（偏航）

`pelvis` body 的坐标系相对世界系绕 x 轴旋转了 -90°，所以**关节名与实际世界方向不一致**：
竖直方向是 ``qpos[1]`` 而不是 ``qpos[0]``。如果只靠关节名写代码很容易搞错（本工作区
曾因此写出错误的测试）。

**为什么骨盆直立轴需要标定**：同理，`pelvis` 的局部 z 轴指向世界 -y（水平），
实测在 ``key_qpos[0]``（全零关节角）下与世界 z 轴夹角为 **90°**。
直接拿 body 的 z 轴当「上」会把任何姿态都判成跌倒，因此用参考姿态标定
（``u_local = R_ref^T @ [0,0,1]``）。

参数来源说明：阈值的**量级**取自对参考轨迹与模型几何的实测（见
``reports/model_inspection_full.json`` 与 ``reports/probe_env.json``），
具体数值写在 `configs/termination_research.json` 里，可显式调整并会被记录进 provenance。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

#: 官方终止规则常量（仅用于对照说明，不改动上游代码）
OFFICIAL_QPOS_DIFF_TH_DEFAULT = 0.06


@dataclass(frozen=True)
class TerminationConfig:
    """研究用终止规则。

    ``kind='official'`` 时完全交回上游 ``LocomotionFullEnvV1.terminated``；
    ``kind='research'`` 时使用本类中的物理条件。
    """

    kind: str = "research"

    #: 骨盆世界系 z 下限（米）。None 表示不检查。
    min_pelvis_height: Optional[float] = 0.55
    #: 骨盆 up 轴与世界 z 轴的最大夹角（度）。None 表示不检查。
    max_pelvis_up_tilt_deg: Optional[float] = 60.0
    #: 根节点平移速度上限（m/s）。
    max_root_speed: Optional[float] = 10.0
    #: 根节点角速度上限（rad/s）。
    max_root_ang_speed: Optional[float] = 50.0
    #: 关节速度范数上限（rad/s 与 m/s 混合，仅作数值爆炸兜底）。
    max_joint_speed_norm: Optional[float] = 200.0
    #: 是否检查 NaN / Inf。
    nan_check: bool = True
    #: 仿真时长上限（秒）。None 时沿用官方的 ``terminate_time * cycles``。
    time_limit_s: Optional[float] = None
    #: 骨盆世界系侧向位置与**当前参考**侧向位置之差的绝对值上限（米）。None 表示不检查。
    #:
    #: 为什么需要这一项（实证依据）：长时失稳诊断显示，人体侧向漂移是**唯一**在跌倒前显著
    #: 发散的信号（``drift_y_human_minus_ref`` 最大 0.438 m，而参考自身摆动只有 0.076 m），
    #: 且它（3.58 s）排在跌倒（4.09 s）之前。但当前密集侧向惩罚只有 −15.8/步，
    #: 而物理存活奖励是 +100/步，相差 6 倍 —— 漂移仍然净赚正回报，靠惩罚改不动。
    #: 改成**终止**后，漂移的代价从「每步 −15.8」变为「损失剩余全部回报」，
    #: 量级上才与存活奖励可竞争。
    #:
    #: 注意：这属于**任务失败**而非生理性跌倒，统计上单独归类（见
    #: ``hemirl.research_env.SOURCE_TASK_LATERAL_DRIFT``），不得混入跌倒计数。
    max_lateral_dev_from_ref_m: Optional[float] = None
    #: 侧向偏移必须**连续**超过 ``max_lateral_dev_from_ref_m`` 的时长（秒）才判为终止。
    #:
    #: 为什么需要持续时长（实测依据，见 ``reports/lateral_threshold_calibration.json``）：
    #: 官方策略的侧向偏移有一个**早期瞬态**（t≈2 s 中位 0.072 m、p90 0.113 m），
    #: 之后回落到 0.012 m，再在 t≈4.5 s 终末发散到 0.30-0.44 m。瞬态与终末发散在
    #: 量级上重叠，所以**单靠位置阈值无法区分“正常步态相位差”与“正在失去平衡”**：
    #: 阈值 0.12 m 时首次越界中位 3.79 s（只比跌倒早 0.69 s），阈值 0.10 m 时
    #: 2.22 s（早 2.71 s，但会打到正常瞬态）。加上持续时长后，可以用较低的阈值
    #: 而不误打在瞬态上。设为 0 或 None 时退化为瞬时判定。
    lateral_hold_s: float = 0.0
    #: 明确声明：**不**因偏离参考姿态而终止。
    allow_reference_deviation_termination: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("official", "research"):
            raise ValueError(f"kind 必须是 'official' 或 'research'，收到 {self.kind!r}")
        if self.allow_reference_deviation_termination:
            raise ValueError(
                "研究模式不支持「因偏离参考姿态而终止」；如需官方行为请设 kind='official'"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TerminationConfig":
        # 允许以 "_comment" 等下划线前缀键写注释
        payload = {k: v for k, v in d.items() if not k.startswith("_")}
        allowed = {f for f in cls.__dataclass_fields__}
        extra = set(payload) - allowed
        if extra:
            raise ValueError(f"TerminationConfig 收到未知字段: {sorted(extra)}")
        return cls(**payload)

    @classmethod
    def from_json(cls, path: Path) -> "TerminationConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


#: 官方复现用的配置（与上游 step() 行为完全一致）
OFFICIAL = TerminationConfig(kind="official", allow_reference_deviation_termination=False)


def pelvis_upright_local_axis(model, ref_qpos: np.ndarray, pelvis_body_id: Optional[int] = None) -> np.ndarray:
    """用参考姿态标定「骨盆直立轴」在骨盆局部坐标系中的方向。

    **为什么需要标定**：MS-Human-700 的 `pelvis` body 局部坐标系并不与解剖学「上」对齐——
    实测在 `key_qpos[0]`（全零关节角）下，pelvis 的局部 z 轴与世界 z 轴夹角为 **90°**。
    若直接拿 body 的 z 轴当「上」，任何姿态都会被判成跌倒。

    标定方式：取参考姿态下的旋转矩阵 ``R_ref``，则
    ``u_local = R_ref^T @ [0,0,1]``，
    即「参考姿态下指向世界的上方」在骨盆局部坐标中的表示。之后任意时刻的直立偏差为
    ``angle(R_cur @ u_local, [0,0,1])``；在参考姿态下该值为 0。

    Args:
        model: MjModel。
        ref_qpos: 参考（直立）姿态的关节配置。
        pelvis_body_id: 骨盆 body 索引；None 时按名字查找。

    Returns:
        长度为 3 的单位向量（骨盆局部坐标系）。
    """
    import mujoco

    if pelvis_body_id is None:
        pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    data = mujoco.MjData(model)
    data.qpos[:] = ref_qpos
    mujoco.mj_forward(model, data)
    rot_ref = np.asarray(data.xmat[pelvis_body_id]).reshape(3, 3)
    u_local = rot_ref.T @ np.array([0.0, 0.0, 1.0])
    return u_local / np.linalg.norm(u_local)


def upright_tilt_deg(rot: np.ndarray, upright_local: Optional[np.ndarray] = None) -> float:
    """骨盆直立偏差角（度）。

    ``upright_local`` 为 :func:`pelvis_upright_local_axis` 的标定结果；
    为 None 时退化为 body 的 z 轴（仅在已知 body 坐标系与「上」对齐时使用）。
    """
    rot = np.asarray(rot).reshape(3, 3)
    axis = np.array([0.0, 0.0, 1.0]) if upright_local is None else np.asarray(upright_local, dtype=float)
    up = rot @ axis
    up = up / max(np.linalg.norm(up), 1e-12)
    cos = float(np.clip(up[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def evaluate_research_termination(
    model,
    data,
    cfg: TerminationConfig,
    pelvis_body_id: int,
    time_s: float,
    upright_local: Optional[np.ndarray] = None,
    ref_y: Optional[float] = None,
    lateral_hold: Optional[float] = None,
) -> Optional[str]:
    """返回终止原因字符串；未触发返回 ``None``。

    Args:
        upright_local: 由 :func:`pelvis_upright_local_axis` 标定的直立轴；用于倾角判据。
        ref_y: 当前参考轨迹的世界系侧向位置（米）。仅在
            ``cfg.max_lateral_dev_from_ref_m`` 非 None 时需要；缺失时**报错**而不是
            静默跳过，否则配置写了却不生效。
        lateral_hold: 调用方维护的「已连续超限时长（秒）」。跨步状态不能在本函数里保存，
            因为同一配置会被多个环境实例共享。
    """
    if cfg.kind == "official":
        return None

    if cfg.nan_check:
        for name, arr in (
            ("qpos", data.qpos),
            ("qvel", data.qvel),
            ("qacc", data.qacc),
            ("act", data.act),
        ):
            if not np.all(np.isfinite(arr)):
                return f"数值异常: {name} 含 NaN/Inf"

    if cfg.min_pelvis_height is not None:
        z = float(data.xpos[pelvis_body_id][2])
        if z < cfg.min_pelvis_height:
            return f"跌到骨盆高度 {z:.3f} m < {cfg.min_pelvis_height} m"

    if cfg.max_pelvis_up_tilt_deg is not None:
        rot = np.asarray(data.xmat[pelvis_body_id]).reshape(3, 3)
        angle_deg = upright_tilt_deg(rot, upright_local)
        if angle_deg > cfg.max_pelvis_up_tilt_deg:
            return f"骨盆直立偏差 {angle_deg:.1f}° > {cfg.max_pelvis_up_tilt_deg}°"

    if cfg.max_root_speed is not None:
        # pelvis_tz/ty/tx 是根节点三个滑动关节，其速度即世界系平移速度
        v = np.asarray(data.qvel[0:3], dtype=float)
        speed = float(np.linalg.norm(v))
        if speed > cfg.max_root_speed:
            return f"根节点平移速度 {speed:.2f} m/s > {cfg.max_root_speed} m/s"

    if cfg.max_root_ang_speed is not None:
        w = np.asarray(data.qvel[3:6], dtype=float)
        omega = float(np.linalg.norm(w))
        if omega > cfg.max_root_ang_speed:
            return f"根节点角速度 {omega:.1f} rad/s > {cfg.max_root_ang_speed} rad/s"

    if cfg.max_joint_speed_norm is not None:
        n = float(np.linalg.norm(np.asarray(data.qvel, dtype=float)))
        if n > cfg.max_joint_speed_norm:
            return f"关节速度范数 {n:.1f} > {cfg.max_joint_speed_norm}"

    if cfg.max_lateral_dev_from_ref_m is not None:
        if ref_y is None:
            raise ValueError(
                "配置要求侧向偏移终止（max_lateral_dev_from_ref_m 非 None），"
                "但调用处未传 ref_y —— 不能静默跳过，否则该终止项看似开启实则无效。"
            )
        y = float(data.xpos[pelvis_body_id][1])
        dev = y - float(ref_y)
        # 持续时长判据由调用方维护（需要跨步状态），此处在 ``lateral_hold`` 参数里传入
        # 已连续超限的时长；未传时按瞬时判定。
        held = float(lateral_hold) if lateral_hold is not None else 0.0
        if abs(dev) > float(cfg.max_lateral_dev_from_ref_m) and held >= float(cfg.lateral_hold_s):
            return (
                f"侧向偏移 {dev:+.3f} m 超过 {cfg.max_lateral_dev_from_ref_m} m "
                f"并已持续 {held:.2f} s（限 {cfg.lateral_hold_s:.2f} s）；"
                f"人体 y={y:+.3f}，参考 y={float(ref_y):+.3f}"
            )

    if cfg.time_limit_s is not None and time_s >= cfg.time_limit_s:
        return f"达到时间上限 {cfg.time_limit_s} s"

    return None


__all__ = [
    "TerminationConfig",
    "OFFICIAL",
    "OFFICIAL_QPOS_DIFF_TH_DEFAULT",
    "evaluate_research_termination",
    "pelvis_upright_local_axis",
    "upright_tilt_deg",
]
