"""环境构建：官方 msgym 环境 + 官方动作包装器。

关键点（不得改动，否则 checkpoint 不再兼容）：

* 环境 ID：``msgym/LocomotionFullEnv-v1``，模型 ``MS-Human-700.xml`` 全身模型。
* 动作空间 = 700 维肌肉 excitation 指令；``MuscleNormWrapper`` 把策略输出
  ``[-1, 1]`` 映射到 ``(0, 1)``。
* 观测空间 3601 维；``VecNormalize``（``norm_obs=True, clip_obs=10``）做归一化。
* ``skip_frames=10``、模型 ``timestep=0.002`` → 控制周期 0.02 s（50 Hz）。
* ``kinematic_play`` 必须为 False（真实动力学）。本模块会强制检查。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from hemirl import paths


@dataclass
class OfficialEnvConfig:
    """官方 checkpoint 训练时的环境配置（原样取自 checkpoint 内 json）。"""

    env_name: str = "msgym/LocomotionFullEnv-v1"
    single_env_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "skip_frames": 10,
            "reset_noise_scale": 0.001,
            "qpos_diff_th": 0.06,
            "gait_cycles": 3,
            "random_init": True,
            "reward_dict": {
                "w_qpos": 50,
                "w_xpos": 50,
                "w_pelvis": 100,
                "w_energy": 0.1,
                "w_healthy": 100,
            },
        }
    )
    wrapper_list: Dict[str, Any] = field(default_factory=lambda: {"MuscleNormWrapper": {}})
    vec_normalize_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"norm_obs": True, "norm_reward": False, "clip_obs": 10.0}
    )
    seed: int = 0

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Path) -> "OfficialEnvConfig":
        """从 checkpoint 目录的 json 读回训练时的环境配置（保证与权重匹配）。

        优先取名为 ``locomotionFull.json`` 的文件（官方 checkpoint 的命名），
        否则取目录下排序最靠前的 ``*.json``。**显式优先**很重要：微调产物目录里还会
        同时存在 ``run.json`` 等元数据文件，只有排序不是「恰好正确」才能保证不读错。
        """
        checkpoint_dir = Path(checkpoint_dir)
        preferred = checkpoint_dir / "locomotionFull.json"
        if preferred.is_file():
            cfg = json.loads(preferred.read_text(encoding="utf-8"))
        else:
            jsons = sorted(checkpoint_dir.glob("*.json"))
            if not jsons:
                raise FileNotFoundError(f"checkpoint 目录下没有配置文件: {checkpoint_dir}")
            cfg = json.loads(jsons[0].read_text(encoding="utf-8"))
        return cls(
            env_name=cfg["env_name"],
            single_env_kwargs=cfg["single_env_kwargs"],
            wrapper_list=cfg["wrapper_list"],
            vec_normalize_kwargs=cfg["vec_normalize"]["kwargs"],
            seed=cfg.get("seed", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_name": self.env_name,
            "single_env_kwargs": self.single_env_kwargs,
            "wrapper_list": self.wrapper_list,
            "vec_normalize_kwargs": self.vec_normalize_kwargs,
            "seed": self.seed,
        }


def build_env(
    cfg: OfficialEnvConfig,
    render_mode: Optional[str] = None,
    kinematic_play: bool = False,
    qpos_diff_th_override: Optional[float] = None,
):
    """创建官方环境并套上官方动作包装器。

    Args:
        cfg: 官方环境配置。
        render_mode: ``None`` / ``"rgb_array"`` / ``"human"`` / ``"depth_array"``。
        kinematic_play: 必须为 False。True 会关闭重力并让状态跟随参考轨迹，
            那是**运动学回放**，不能用于动力学研究；本函数会拒绝 True。
        qpos_diff_th_override: 覆盖官方的姿态偏差阈值（一般不使用）。

    Returns:
        (wrapped_env, raw_env)
    """
    if kinematic_play:
        raise ValueError(
            "kinematic_play=True 是运动学回放（重力置零 + 状态跟随参考），"
            "不得用于本研究的动力学评估。请保持 False。"
        )
    paths.ensure_msgym_on_path()
    paths.ensure_msgym_model_link()
    import gymnasium as gym
    import msgym  # noqa: F401,F811

    from hemirl.wrappers import official_muscle_norm_wrapper

    kwargs = dict(cfg.single_env_kwargs)
    if qpos_diff_th_override is not None:
        kwargs["qpos_diff_th"] = qpos_diff_th_override
    env = gym.make(cfg.env_name, render_mode=render_mode, **kwargs)

    raw = env
    wrappers = list(cfg.wrapper_list)
    known = {"MuscleNormWrapper": official_muscle_norm_wrapper}
    for name in wrappers:
        if name not in known:
            raise KeyError(f"不支持的包装器 {name!r}；官方配置只允许 {sorted(known)}")
        env = known[name]()(env)

    # 强制检查：真实动力学 + 动作/观测维度与 checkpoint 一致
    if bool(getattr(raw.unwrapped, "kinematic_play", False)):
        raise RuntimeError("环境 kinematic_play 为 True，属于运动学回放，已中止")
    return env, raw


__all__ = ["OfficialEnvConfig", "build_env"]
