"""研究环境：训练与评估共用的终止 / 计时 / 奖励语义。

## 为什么需要一个独立环境层

上游 ``LocomotionFullEnvV1`` 的语义不适合本研究：

============  ====================================================  ==========================
问题           上游行为                                               后果
============  ====================================================  ==========================
语义混用       ``terminated = not is_healthy or t >= T*cycles``       参考姿态偏差与「时间到」被
               ``truncated  = t >= T*cycles``                         压进同一个 terminated 字段
参考偏差终止    偏离参考姿态即终止                                    这不是「平衡失败」，而是
                                                                     「没有跟住某个录像」
时间上限被写死  上限 = ``terminate_time * cycles`` = 3.51 s            无法做长时评估
最后一步         在 ``step`` 内部即已算完 terminated，但调用方常在     终止瞬间的物理状态丢失
               记录前 break
奖励含参考项     ``w_healthy`` 项 = ``is_healthy``（参考姿态依赖）      物理存活与模仿混在一起，
                                                                     训练出的策略含义不清
============  ====================================================  ==========================

本模块把上述语义**显式拆开**，并且**不改动上游代码**（只在外层包装）：

* ``terminated``：物理跌倒（骨盆过低 / 姿态倾覆）或数值异常；
* ``truncated``：达到规定的评估时长，或达到本地控制步上限；
* 数值异常：``termination_source='numeric_anomaly'``，与生理性跌倒分开统计；
* 时间由**实际仿真时间差**给出（``data.time`` 的差），而不是步数乘以标称 dt；
* 每一步（含终止步）都写进 ledger；
* 终止后继续 ``step`` 直接抛错，不会静默推进、也不会自动 reset；
* 奖励拆成 imitation / energy / survival_physical / official_healthy 四项分别返回。

## 接口兼容

观测与动作接口**完全不变**：本环境包装的是官方 ``MuscleNormWrapper`` 之后的 env，
不新增任何观测分量。需要新观测时请在**独立的后续配置**里做（见 ``reward_mode='split'``
仅影响奖励与 info，不影响 obs/action 维度）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from hemirl import paths
from hemirl.termination import (
    TerminationConfig,
    evaluate_research_termination,
    pelvis_upright_local_axis,
    upright_tilt_deg,
)

#: 终止来源标记（落盘用，避免把不同性质的结束混在一起）
SOURCE_PHYSICAL_FALL = "physical_fall"
SOURCE_NUMERIC_ANOMALY = "numeric_anomaly"
SOURCE_TIME_LIMIT = "time_limit"
SOURCE_STEP_CAP = "step_cap"
SOURCE_OFFICIAL_REFERENCE = "official_reference_deviation"
SOURCE_OFFICIAL_TIME = "official_time_limit"

#: 奖励模式
REWARD_MODE_OFFICIAL = "official"  # 返回上游 reward（与 checkpoint 训练时一致）
REWARD_MODE_SPLIT = "split"  # 返回物理存活 + 模仿（供后续损伤适应训练使用）
REWARD_MODES = (REWARD_MODE_OFFICIAL, REWARD_MODE_SPLIT)


@dataclass
class RewardSplitConfig:
    """奖励拆分配置（研究中「物理存活」与「模仿」必须分开）。

    上游 ``w_healthy`` 项 = ``is_healthy``，而 ``is_healthy`` 是**参考姿态偏差**的函数，
    因此上游奖励里并不存在一个「与任务无关的物理存活」项。本配置把它显式补上：

    * ``imitation``：``w_qpos·r_qpos + w_xpos·r_xpos + w_pelvis·r_pelvis``（全部依赖参考）；
    * ``energy``：``w_energy·r_energy``（不依赖参考）；
    * ``survival_physical``：骨盆高于 ``min_pelvis_height`` 且无数值异常则给 ``w_survival``，
      否则 0。**不依赖参考轨迹**；
    * ``official_healthy``：上游那一项，单独保留以便对齐官方语义。

    Attributes:
        w_survival: 物理存活项权重（仅 ``reward_mode='split'`` 时进入返回值）。
        include_official_healthy: ``reward_mode='split'`` 时是否把上游 ``w_healthy`` 项
            计入返回值。默认 False——它依赖参考姿态，属于模仿项。
        normalization: 可选的整体缩放，便于与官方奖励量级对比。默认 1.0。
    """

    w_survival: float = 100.0
    include_official_healthy: bool = False
    normalization: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchEnvConfig:
    """研究环境配置。"""

    termination: TerminationConfig = field(default_factory=TerminationConfig)
    #: 规定的评估/训练时长（秒）。研究模式下用于 truncated；None 表示沿用官方 3.51 s。
    max_episode_seconds: Optional[float] = None
    #: 本地控制步上限；None 时由 ``max_episode_seconds / dt`` 推出。
    max_control_steps: Optional[int] = None
    #: 奖励返回模式。
    reward_mode: str = REWARD_MODE_OFFICIAL
    reward_split: RewardSplitConfig = field(default_factory=RewardSplitConfig)
    #: 是否保留逐步 ledger（长回合内存开销与步数成正比，默认开启，训练时可关）。
    keep_ledger: bool = True
    #: 便于溯源的名字。
    name: str = "research"

    def __post_init__(self) -> None:
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode 必须是 {REWARD_MODES}，收到 {self.reward_mode!r}")
        if self.termination.kind not in ("official", "research"):
            raise ValueError(f"termination.kind 非法: {self.termination.kind!r}")

    @property
    def suppresses_official_reference_termination(self) -> bool:
        """研究模式必须覆盖官方的「参考姿态偏差」终止。"""
        return self.termination.kind == "research"

    @property
    def suppresses_official_time_truncation(self) -> bool:
        """研究模式必须覆盖官方写死的时间截断（改由 max_episode_seconds 决定）。"""
        return self.termination.kind == "research"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "termination": self.termination.to_dict(),
            "max_episode_seconds": self.max_episode_seconds,
            "max_control_steps": self.max_control_steps,
            "reward_mode": self.reward_mode,
            "reward_split": self.reward_split.to_dict(),
            "keep_ledger": self.keep_ledger,
            "suppresses_official_reference_termination": self.suppresses_official_reference_termination,
            "suppresses_official_time_truncation": self.suppresses_official_time_truncation,
        }

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: Path) -> "ResearchEnvConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = {k: v for k, v in payload.items() if not k.startswith("_")}
        term = payload.pop("termination", None)
        rsplit = payload.pop("reward_split", None)
        cfg = cls(**payload)
        if term is not None:
            cfg.termination = TerminationConfig.from_dict(term)
        if rsplit is not None:
            cfg.reward_split = RewardSplitConfig(**rsplit)
        return cfg


@dataclass
class StepRecord:
    """一步（控制步）的完整记录。**包含触发终止的那一步**。"""

    index: int
    sim_time_before: float
    sim_time_after: float
    control_dt: float
    physics_steps: int
    reward_total: float
    reward_imitation: float
    reward_energy: float
    reward_survival_physical: float
    reward_official_healthy: float
    qpos_track_err: float
    pelvis_z: float
    up_tilt_deg: float
    official_terminated_flag: bool
    official_truncated_flag: bool
    termination_reason: Optional[str] = None
    termination_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchLocomotionEnv(gym.Wrapper):
    """在官方环境之外显式实现研究语义的包装器。

    继承 ``gymnasium.Wrapper``（而不是自建一个鸭子类型对象），因为
    ``stable_baselines3.DummyVecEnv`` 会做 ``isinstance(env, gym.Env)`` 检查——
    载入官方 ``VecNormalize`` 时必须能通过该检查。

    ``self.raw_env`` 显式保存 ``unwrapped`` 出来的上游环境：gymnasium 1.2 的 ``Wrapper``
    **不再**做通用属性转发（只显式转发 ``render_mode`` / ``metadata`` / ``action_space`` /
    ``observation_space`` / ``unwrapped``），所以要拿 ``model`` / ``data`` / ``dt`` /
    ``qpos_ref`` 等必须走 ``unwrapped``。
    """

    def __init__(
        self,
        env,
        cfg: Optional[ResearchEnvConfig] = None,
        raw_env=None,
    ):
        super().__init__(env)
        self.cfg = cfg or ResearchEnvConfig()
        self.raw_env = (raw_env if raw_env is not None else env).unwrapped

        model = self.raw_env.model
        data = self.raw_env.data
        self.model = model
        self.data = data
        import mujoco

        self.pelvis_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
        self.dt = float(self.raw_env.dt)
        self.frame_skip = int(self.raw_env.frame_skip)
        self.physics_timestep = float(model.opt.timestep)
        self.nu = int(model.nu)

        # 官方写死的时间上限（仅用于「官方模式保留原行为」）
        cycles = int(getattr(self.raw_env, "cycles", 1))
        self.official_time_limit_s = float(self.raw_env.terminate_time) * cycles

        # 研究模式的时间上限
        if self.cfg.termination.kind == "official":
            self.time_limit_s: Optional[float] = None
        elif self.cfg.max_episode_seconds is not None:
            self.time_limit_s = float(self.cfg.max_episode_seconds)
        else:
            self.time_limit_s = self.official_time_limit_s

        # 步数上限**向上取整**：``3.51/0.02`` 在浮点下是 175.49999...，用 round 会得到 175，
        # 于是「步数上限」比「时间上限」早一步触发，改变评估口径（官方模式是 176 步 / 3.52 s）。
        # 向上取整后时间上限才是真正的约束，步数上限只是防无限循环的兜底。
        if self.cfg.max_control_steps is not None:
            self.max_control_steps = int(self.cfg.max_control_steps)
        elif self.time_limit_s is not None:
            self.max_control_steps = int(np.ceil(round(self.time_limit_s / self.dt, 9)))
        else:
            self.max_control_steps = int(np.ceil(round(self.official_time_limit_s / self.dt, 9)))

        # 运行状态
        self.ledger: List[StepRecord] = []
        self._reset_time = 0.0
        self._upright_local: Optional[np.ndarray] = None
        self._done = False
        self._termination_reason: Optional[str] = None
        self._termination_source: Optional[str] = None
        self._terminated = False
        self._truncated = False
        self._n_official_term_flags = 0
        self._n_steps_executed = 0
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------ 兼容属性

    def action(self, action):
        """策略动作 → 肌肉 excitation（转发官方 ``MuscleNormWrapper.action``）。"""
        return self.env.action(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        self.env.close()
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def configure_termination(self, term: TerminationConfig) -> None:
        """替换终止配置并**重算**由它派生的时间上限 / 步数上限。

        派生量在 ``__init__`` 里算好后不会再变，因此直接改 ``cfg.termination`` 会让
        ``max_control_steps`` 与 ``time_limit_s`` 变成陈旧值（例如官方模式报 3.51 s
        但研究步数上限还是 20 s）。本方法保证两者始终一致。
        """
        self.cfg = replace(self.cfg, termination=term)
        if term.kind == "official":
            self.time_limit_s = None
        elif self.cfg.max_episode_seconds is not None:
            self.time_limit_s = float(self.cfg.max_episode_seconds)
        else:
            self.time_limit_s = self.official_time_limit_s
        if self.cfg.max_control_steps is not None:
            self.max_control_steps = int(self.cfg.max_control_steps)
        elif self.time_limit_s is not None:
            self.max_control_steps = int(np.ceil(round(self.time_limit_s / self.dt, 9)))
        else:
            self.max_control_steps = int(np.ceil(round(self.official_time_limit_s / self.dt, 9)))

    # ------------------------------------------------------------ 状态

    @property
    def done(self) -> bool:
        return self._done

    @property
    def n_control_steps_executed(self) -> int:
        return len(self.ledger) if self.cfg.keep_ledger else self._n_steps_executed

    @property
    def elapsed_time_s(self) -> float:
        """实际仿真时间差（不是步数 × 标称 dt）。"""
        return float(self.data.time - self._reset_time)

    @property
    def n_physics_steps_executed(self) -> int:
        return self.n_control_steps_executed * self.frame_skip

    def termination_metadata(self) -> Dict[str, Any]:
        """显式终止元数据（任务书要求：reason / source / elapsed / 实际控制步数）。"""
        return {
            "termination_reason": self._termination_reason,
            "termination_source": self._termination_source,
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
            "elapsed_time_s": self.elapsed_time_s,
            "n_control_steps": self.n_control_steps_executed,
            "n_physics_steps": self.n_physics_steps_executed,
            "n_official_terminated_flags": self._n_official_term_flags,
            "numeric_anomaly": self._termination_source == SOURCE_NUMERIC_ANOMALY,
            "physical_fall": self._termination_source == SOURCE_PHYSICAL_FALL,
            # 明确的截断标志（任务书第 5 条：达到本地上限必须能被显式识别）
            "reached_time_limit": self._termination_source == SOURCE_TIME_LIMIT,
            "reached_step_cap": self._termination_source == SOURCE_STEP_CAP,
            "reached_official_time_limit": self._termination_source == SOURCE_OFFICIAL_TIME,
            "time_limit_s": self.time_limit_s,
            "max_control_steps": self.max_control_steps,
            "official_time_limit_s": self.official_time_limit_s,
            "suppressed_official_reference_termination": self.cfg.suppresses_official_reference_termination,
            "suppressed_official_time_truncation": self.cfg.suppresses_official_time_truncation,
        }

    # ------------------------------------------------------------ reset / step

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.ledger = []
        self._n_steps_executed = 0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._termination_reason = None
        self._termination_source = None
        self._n_official_term_flags = 0
        self._reset_time = float(self.data.time)
        # 按本回合参考姿态标定骨盆直立轴（pelvis body 局部坐标系与世界「上」不对齐）
        self._upright_local = pelvis_upright_local_axis(
            self.model, np.asarray(self.raw_env.qpos_ref), self.pelvis_id
        )
        info = dict(info or {})
        info["research_reset"] = {
            "reset_time_s": self._reset_time,
            "time_limit_s": self.time_limit_s,
            "max_control_steps": self.max_control_steps,
            "termination_kind": self.cfg.termination.kind,
            "reward_mode": self.cfg.reward_mode,
        }
        self._last_info = info
        return obs, info

    def step(self, action):
        """执行一步；返回 ``(obs, reward, terminated, truncated, info)``。

        Raises:
            RuntimeError: 环境已终止仍被继续推进（对应任务书第 6 条）。
        """
        if self._done:
            raise RuntimeError(
                "环境已终止，不允许继续推进（避免在终止后静默推进或自动 reset）；"
                "如需新一轮请显式调用 reset()。"
            )

        t_before = float(self.data.time)
        obs, reward_official, term_official, trunc_official, info = self.env.step(action)
        t_after = float(self.data.time)

        if bool(term_official):
            self._n_official_term_flags += 1

        info = dict(info or {})
        components = self._reward_components(info)
        reason, source = self._evaluate_termination(t_after, term_official, trunc_official)

        index = self._n_steps_executed
        self._n_steps_executed += 1

        if self.cfg.keep_ledger:
            self.ledger.append(
                StepRecord(
                    index=index,
                    sim_time_before=t_before,
                    sim_time_after=t_after,
                    control_dt=self.dt,
                    physics_steps=self.frame_skip,
                    reward_total=float(reward_official),
                    reward_imitation=components["imitation"],
                    reward_energy=components["energy"],
                    reward_survival_physical=components["survival_physical"],
                    reward_official_healthy=components["official_healthy"],
                    qpos_track_err=self.reference_track_err(),
                    pelvis_z=self.pelvis_height(),
                    up_tilt_deg=self.up_tilt_deg(),
                    official_terminated_flag=bool(term_official),
                    official_truncated_flag=bool(trunc_official),
                    termination_reason=reason,
                    termination_source=source,
                )
            )

        # 研究模式：返回值取研究语义；官方模式：原样保留上游行为
        if self.cfg.termination.kind == "official":
            terminated = bool(term_official)
            truncated = bool(trunc_official)
            # 必须把结果写回内部状态，否则 termination_metadata() 会报告「未终止」
            self._terminated = terminated
            self._truncated = truncated
            self._termination_reason = reason
            self._termination_source = source
        else:
            terminated = self._terminated
            truncated = self._truncated

        self._done = bool(terminated or truncated)
        info["termination"] = self.termination_metadata()
        info["reward_components"] = components
        info["reward_split_config"] = self.cfg.reward_split.to_dict() if self.cfg.reward_mode == REWARD_MODE_SPLIT else None
        info["policy_interface_note"] = (
            "obs/action 维度与官方 checkpoint 完全一致；reward_mode 只影响奖励与 info"
        )

        reward = self._select_reward(reward_official, components)
        self._last_info = info
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ 内部

    def _select_reward(self, reward_official: float, components: Dict[str, float]) -> float:
        if self.cfg.reward_mode == REWARD_MODE_OFFICIAL:
            return float(reward_official)
        total = components["imitation"] + components["energy"] + components["survival_physical"]
        if self.cfg.reward_split.include_official_healthy:
            total += components["official_healthy"]
        return float(total * self.cfg.reward_split.normalization)

    def _reward_components(self, info: Dict[str, Any]) -> Dict[str, float]:
        """把上游奖励拆成四项。上游 ``info`` 已给出逐项分量。

        ``reward_healthy`` 是上游的参考姿态依赖项（``is_healthy``），归入
        ``official_healthy``，**不**计入物理存活项。
        """
        def get(key: str) -> float:
            v = info.get(key)
            return float(v) if v is not None else float("nan")

        imitation = get("reward_qpos") + get("reward_xpos") + get("reward_pelvis")
        energy = get("reward_energy")
        official_healthy = get("reward_healthy")
        # 物理存活：只看骨盆高度与数值有限性，不看参考轨迹
        physically_alive = self._is_physically_alive()
        survival = float(self.cfg.reward_split.w_survival) if physically_alive else 0.0
        return {
            "imitation": float(imitation),
            "energy": float(energy),
            "official_healthy": float(official_healthy),
            "survival_physical": survival,
            "survival_physical_is_reference_dependent": False,
            "official_healthy_is_reference_dependent": True,
        }

    def _is_physically_alive(self) -> bool:
        for arr in (self.data.qpos, self.data.qvel, self.data.qacc):
            if not np.all(np.isfinite(arr)):
                return False
        if self.cfg.termination.min_pelvis_height is not None:
            if self.pelvis_height() < float(self.cfg.termination.min_pelvis_height):
                return False
        return True

    def _evaluate_termination(
        self,
        sim_time: float,
        term_official: bool,
        trunc_official: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        """研究模式的终止判定；返回 ``(reason, source)``。

        研究模式必须**覆盖**两类官方行为：

        * 官方参考姿态偏差终止 → 不作为终止依据（只作为 ``official_terminated_flags`` 记录）；
        * 官方写死的时间截断 → 由 ``max_episode_seconds`` 取代。

        数值异常单独归类，不混入生理性跌倒。物理判据全部委托给
        :func:`hemirl.termination.evaluate_research_termination`（传入``time_limit_s=None``
        的子配置），以保证「研究终止规则」只有一处实现，不会在评估循环里漂移。
        """
        if self.cfg.termination.kind == "official":
            # 官方模式：把上游行为原样转成 (reason, source)
            if trunc_official or (term_official and sim_time >= self.official_time_limit_s):
                return ("官方规则: 达到官方时间上限", SOURCE_OFFICIAL_TIME)
            if term_official:
                return ("官方规则: is_healthy 为假（偏离参考姿态超阈值）", SOURCE_OFFICIAL_REFERENCE)
            return (None, None)

        # 1) 数值异常：单独归类，不混入生理性跌倒统计
        for name, arr in (
            ("qpos", self.data.qpos),
            ("qvel", self.data.qvel),
            ("qacc", self.data.qacc),
            ("act", self.data.act),
        ):
            if not np.all(np.isfinite(arr)):
                reason = f"数值异常: {name} 含 NaN/Inf"
                self._terminated, self._termination_reason, self._termination_source = (
                    True,
                    reason,
                    SOURCE_NUMERIC_ANOMALY,
                )
                return (reason, SOURCE_NUMERIC_ANOMALY)

        # 2) 物理跌倒 / 明确任务失败（不含参考姿态偏差，不含时间上限）
        phys_cfg = replace(self.cfg.termination, time_limit_s=None)
        r = evaluate_research_termination(
            self.model,
            self.data,
            phys_cfg,
            self.pelvis_id,
            sim_time,
            upright_local=self._upright_local,
        )
        if r is not None:
            src = SOURCE_NUMERIC_ANOMALY if r.startswith("数值异常") else SOURCE_PHYSICAL_FALL
            self._terminated, self._termination_reason, self._termination_source = True, r, src
            return (r, src)

        # 3) 达到规定评估时长 → truncated（不是 terminated）
        if self.time_limit_s is not None and sim_time >= self.time_limit_s:
            reason = f"达到规定评估时长 {self.time_limit_s} s"
            self._truncated, self._termination_reason, self._termination_source = (
                True,
                reason,
                SOURCE_TIME_LIMIT,
            )
            return (reason, SOURCE_TIME_LIMIT)

        # 4) 本地控制步上限 → truncated，且必须有明确的截断标志
        if self._n_steps_executed + 1 >= self.max_control_steps:
            reason = f"达到本地控制步上限 {self.max_control_steps} 步"
            self._truncated, self._termination_reason, self._termination_source = (
                True,
                reason,
                SOURCE_STEP_CAP,
            )
            return (reason, SOURCE_STEP_CAP)

        return (None, None)

    # ------------------------------------------------------------ 指标

    def pelvis_height(self) -> float:
        return float(self.data.xpos[self.pelvis_id][2])

    def up_tilt_deg(self) -> float:
        rot = np.asarray(self.data.xmat[self.pelvis_id]).reshape(3, 3)
        return upright_tilt_deg(rot, self._upright_local)

    def reference_track_err(self) -> float:
        """参考跟踪误差（**描述性指标**，不作为终止依据，也不代表平衡能力）。"""
        qref = np.asarray(self.raw_env.qpos_ref)
        return float(np.mean(np.abs(np.asarray(self.data.qpos)[3:] - qref[3:])))

    def is_healthy_official(self) -> bool:
        return bool(self.raw_env.is_healthy)

    def ledger_to_dict(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.ledger]

    def accounting_report(self) -> Dict[str, Any]:
        """核对「每一步都被记录」与「运行时间来自实际仿真时间差」。"""
        n = self.n_control_steps_executed
        elapsed = self.elapsed_time_s
        nominal = n * self.dt
        gaps = [r.sim_time_after - r.sim_time_before for r in self.ledger]
        expected_gap = self.frame_skip * self.physics_timestep
        return {
            "n_control_steps_recorded": len(self.ledger),
            "n_control_steps_executed": n,
            "all_steps_recorded": len(self.ledger) == n,
            "n_physics_steps": n * self.frame_skip,
            "physics_step_size_s": self.physics_timestep,
            "control_dt_s": self.dt,
            "elapsed_time_s": elapsed,
            "nominal_time_s": nominal,
            "time_source": "data.time 差（实际仿真时间），非步数×dt",
            "max_abs_diff_elapsed_vs_nominal": abs(elapsed - nominal),
            "per_step_gap_max_dev": float(np.max(np.abs(np.asarray(gaps) - expected_gap))) if gaps else None,
            "last_step_recorded": bool(self.ledger) and self.ledger[-1].termination_reason is not None,
            "last_step_termination_source": self.ledger[-1].termination_source if self.ledger else None,
            "ledger_covers_terminating_step": bool(self.ledger)
            and self.ledger[-1].index == n - 1,
        }


# ------------------------------------------------------------------ 工厂


def build_research_env(
    env_config=None,
    env_cfg: Optional[ResearchEnvConfig] = None,
    render_mode: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
):
    """统一的入口：官方环境 + 研究语义包装。

    **训练与评估都应当走这个函数**，以保证两边的终止/计时/奖励语义完全相同。

    Args:
        env_config: ``hemirl.envs.OfficialEnvConfig``；None 时从 checkpoint 读回。
        env_cfg: 研究环境配置；None 时用默认（研究终止规则 + 官方奖励）。
        render_mode: 传给官方环境。
        checkpoint_dir: 读取官方环境配置的位置。

    Returns:
        ``(research_env, upstream_env)``。第二个元素是 ``unwrapped`` 后的上游
        ``LocomotionFullEnvV1``（不是 ``gym.make`` 返回的包裹链），因为 gymnasium 1.2 的
        ``Wrapper`` 不再做通用属性转发，拿到包裹对象也取不到 ``model`` / ``qpos_ref``。
    """
    from hemirl.envs import OfficialEnvConfig, build_env

    if env_config is None:
        ckpt = Path(checkpoint_dir) if checkpoint_dir else paths.checkpoint_dir("LocomotionFull")
        env_config = OfficialEnvConfig.from_checkpoint(ckpt)
    base, raw = build_env(env_config, render_mode=render_mode)
    research = ResearchLocomotionEnv(base, env_cfg or ResearchEnvConfig(), raw_env=raw)
    return research, research.raw_env


# ------------------------------------------------------- 参考轨迹连续性检查


def reference_continuity_report(
    raw_env,
    n_cycles: int = 20,
    dt: Optional[float] = None,
) -> Dict[str, Any]:
    """检查参考轨迹的循环、前进位移累积与未来参考观测是否连续。

    必须在长时评估**之前**做：上游 ``LocomotionCycleTrajectory.query_batch`` 用

        qpos[:, 0] += cycle_number * qpos_traj[-1, 0]
        qpos[:, 2] += cycle_number * qpos_traj[-1, 2]

    做跨周期位移累积，其中 ``cycle_number = floor(t / terminate_time)`` 而
    ``terminate_time = num_frames / framerate``。注意 ``time_step`` 被
    ``clip(..., 0, num_frames-1)`` 截断，因此帧序列只能覆盖 ``(num_frames-1)/framerate``，
    比 ``terminate_time`` **短一帧**；这一帧的缺口会在每个周期边界变成一次跳变。
    如果 ``qpos_traj[-1]`` 的关节角不等于 ``qpos_traj[0]``（轨迹非严格周期），
    缺口还会叠加姿态不连续。

    本函数按**控制步长**采样参考（即评估时真实看到的序列），把「周期内单步变化」
    与「周期边界单步变化」分开报告，并给出累计效应，供长时评估解释。

    Returns:
        含 ``within_cycle`` / ``at_boundary`` / ``pose_wrap`` / ``forward_*`` /
        ``future_*`` / ``needs_loop_guard`` 等字段的字典。
    """
    traj = raw_env.trajectory
    idx = int(getattr(raw_env, "current_traj_index", 0))
    terminate_time, velocity, stride = traj.get_trajectory_properties(idx)
    step = float(dt if dt is not None else raw_env.dt)
    t = traj.trajectories[idx]
    qp = np.asarray(t["qpos_traj"], dtype=float)
    num_frames = int(t["num_frames"])
    framerate = float(t["framerate"])

    # --- 按控制步长采样参考序列（评估时真实见到的序列）
    n_samples = int(np.ceil(n_cycles * terminate_time / step))
    times = np.arange(n_samples, dtype=np.float64) * step
    qpos, _xpos, _qvel = traj.query_batch(times, idx)
    cycle_number = np.floor(times / terminate_time).astype(np.int64)

    diffs = np.diff(qpos, axis=0)
    step_mag = np.max(np.abs(diffs), axis=1)
    boundary_mask = cycle_number[1:] != cycle_number[:-1]
    within = step_mag[~boundary_mask]
    at_boundary = step_mag[boundary_mask]

    # --- 帧序列本身的属性（解释跳变来源）
    wrap_frame_gap = float(terminate_time - (num_frames - 1) / framerate)
    pose_wrap = float(np.max(np.abs(qp[-1][3:] - qp[0][3:])))
    per_frame = float(np.max(np.abs(np.diff(qp, axis=0))))

    # --- 前进位移累积
    per_cycle_qpos_x = float(qp[-1, 2] - qp[0, 2])  # pelvis_tx 分量的周期增量
    per_cycle_xpos_x = float(t["pelvis_trans"][0])
    pelvis = int(raw_env.pelvis_id)
    xpos_forward = np.asarray(_xpos[:, pelvis, 0], dtype=float)
    achieved_forward = float(xpos_forward[-1] - xpos_forward[0])
    expected_forward = per_cycle_qpos_x * (n_samples - 1) * step / terminate_time

    # --- 未来参考观测（与上游 step 内一致：ref_time = time + init_time + dt + k*dt）
    future_steps = int(getattr(raw_env, "future_traj_steps", 5))
    ref_times = step + np.arange(future_steps + 1, dtype=np.float64) * step
    fut, _fx, _fv = traj.query_batch(ref_times, idx)
    fut_step = float(np.max(np.abs(np.diff(fut, axis=0))))

    boundary_ratio = (float(np.max(at_boundary)) / float(np.max(within))) if within.size and at_boundary.size else None
    return {
        "traj_index": idx,
        "terminate_time_s": float(terminate_time),
        "velocity_mps": float(velocity),
        "stride_m": float(stride),
        "framerate_hz": framerate,
        "num_frames": num_frames,
        "frame_span_s": float((num_frames - 1) / framerate),
        "control_step_s": step,
        "n_cycles_checked": int(n_cycles),
        "n_reference_samples": int(n_samples),
        "n_cycle_boundaries": int(boundary_mask.sum()),

        # 连续性的核心对照
        "max_step_change_within_cycle": float(np.max(within)) if within.size else None,
        "max_step_change_at_cycle_boundary": float(np.max(at_boundary)) if at_boundary.size else None,
        "max_step_change_mean_within_cycle": float(np.mean(within)) if within.size else None,
        "boundary_over_within_ratio": boundary_ratio,
        "boundary_is_comparable_to_one_step": bool(boundary_ratio is not None and boundary_ratio <= 2.0),

        # 跳变的来源
        "wrap_frame_gap_s": wrap_frame_gap,
        "pose_wrap_discontinuity_max_rad": pose_wrap,
        "per_frame_change_max_within_cycle": per_frame,
        "cycle_is_strictly_periodic": bool(pose_wrap <= 1e-9),

        # 前进位移累积
        "per_cycle_forward_from_qpos_m": per_cycle_qpos_x,
        "per_cycle_forward_from_xpos_m": per_cycle_xpos_x,
        "per_cycle_forward_qpos_vs_xpos_diff_m": abs(per_cycle_qpos_x - per_cycle_xpos_x),
        "stride_metadata_vs_per_cycle_qpos_diff_m": abs(float(stride) - per_cycle_qpos_x),
        "achieved_forward_m": achieved_forward,
        "expected_forward_from_per_cycle_m": expected_forward,
        "forward_accumulation_rel_err": abs(achieved_forward / max(1e-12, expected_forward) - 1.0),

        # 未来参考观测
        "future_traj_steps": future_steps,
        "future_ref_step_change_max": fut_step,
        "reference_time_offset_note": (
            "上游用 ref_time = data.time + init_time + dt，因此参考相对当前状态超前 1 个控制步"
            f"（{step:.3f} s）。这是上游既有行为，为保持 checkpoint 接口一致不作修改。"
        ),

        # 累计效应用于解释长时评估
        "cumulative_pose_wrap_over_eval_rad": pose_wrap * float(n_cycles),
        "cumulative_forward_gap_over_eval_m": per_cycle_qpos_x * wrap_frame_gap / terminate_time * float(n_cycles),
        "needs_loop_guard": bool(
            pose_wrap > 10 * per_frame or (boundary_ratio is not None and boundary_ratio > 2.0)
        ),
        "interpretation": {
            "within_cycle": "max_step_change_within_cycle 应与一个控制步内的正常运动量同阶",
            "boundary": "max_step_change_at_cycle_boundary 若与 within 同阶，则循环处无可感知跳变",
            "wrap_gap": "wrap_frame_gap_s>0 表示帧序列比 terminate_time 短一帧，边界必然有一次跳变",
            "pose_wrap": "pose_wrap_discontinuity_max_rad>0 表示轨迹非严格周期，姿态在边界不闭合",
        },
    }


__all__ = [
    "ResearchEnvConfig",
    "RewardSplitConfig",
    "ResearchLocomotionEnv",
    "StepRecord",
    "build_research_env",
    "reference_continuity_report",
    "SOURCE_PHYSICAL_FALL",
    "SOURCE_NUMERIC_ANOMALY",
    "SOURCE_TIME_LIMIT",
    "SOURCE_STEP_CAP",
    "SOURCE_OFFICIAL_REFERENCE",
    "SOURCE_OFFICIAL_TIME",
    "REWARD_MODE_OFFICIAL",
    "REWARD_MODE_SPLIT",
]
