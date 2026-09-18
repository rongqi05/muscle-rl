"""官方 DynSyn-SAC checkpoint 加载与推理，两条独立路径互验。

## 路径 A：官方 SB3 路径

复用上游 ``DynSyn/SAC_DynSyn.py`` 的 ``Actor_DynSyn`` / ``DynSynLayer``，
用 ``stable_baselines3`` 的 ``load`` 读 ``best_model.zip``，再用 ``VecNormalize``
读 ``checkpoint/best_env.zip`` 做观测归一化（与官方 ``eval.py`` 一致）。

## 路径 B：独立 torch 直读路径

`best_model.zip` 里的 ``policy.pth`` 就是普通 torch state_dict（键见下）。
本模块用纯 torch 重建 actor 前向，作为对路径 A 的**独立交叉验证**：
两条路径在 ``deterministic=True`` 时的输出必须逐元素一致。

已知网络结构（由 state_dict 键与形状实证）::

    actor.latent_pi.0  Linear(3601 -> 1024)   + ReLU
    actor.latent_pi.2  Linear(1024 -> 1024)   + ReLU
    actor.latent_pi.4  Linear(1024 -> 1024)
    actor.mu           Linear(1024 -> 138)     # 138 = DynSyn 肌群数
    actor.log_std      Linear(1024 -> 138)
    actor.dynsyn_layer.mu.0 Linear(1024 -> 562)  # dynsyn_weight_amp=None 时输出被丢弃

## dynsyn_weight_amp 的评估时取值

checkpoint 的 ``data`` 中 ``dynsyn_weight_amp`` 为 **null**，而 ``DynSynLayer.forward``
在 ``dynsyn_weight_amp is None`` 时把 ``weight`` 置为 1，因此评估时：动作 = 肌群动作
按组展开（``repeat_replace_x``），不做协同权重重标定。两条路径都按此语义实现。
"""

from __future__ import annotations

import contextlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from hemirl import paths
from hemirl.numpy_compat import compat_pickle_load, install_numpy2_pickle_compat

#: 载入 checkpoint 前必须安装：checkpoint 由 numpy 2.x 序列化，本环境是 numpy 1.26
install_numpy2_pickle_compat()

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# ------------------------------------------------------------------ 组展开


def build_group_expansion(muscle_groups: Sequence[Sequence[int]]) -> Dict[str, Any]:
    """复刻 `DynSynLayer.repeat_replace_x` 的索引语义。

    返回：
        ``muscle_dims``：覆盖的肌肉总数（应为 700）。
        ``group_of_muscle``：长度 ``muscle_dims`` 的数组，第 j 项是肌肉 j 所属组的编号。
    """
    muscle_dims = max(max(g) for g in muscle_groups) + 1
    group_nums = len(muscle_groups)
    group_of = np.full(muscle_dims, -1, dtype=np.int64)
    for gi, group in enumerate(muscle_groups):
        for j in group:
            if group_of[j] != -1:
                raise ValueError(f"肌肉 {j} 出现在多个组中（{group_of[j]} 与 {gi}）")
            group_of[j] = gi
    if (group_of < 0).any():
        missing = np.where(group_of < 0)[0]
        raise ValueError(f"以下肌肉不属于任何组: {missing[:20].tolist()} (共 {missing.size})")
    return {
        "muscle_dims": int(muscle_dims),
        "muscle_group_nums": int(group_nums),
        "group_of_muscle": group_of,
    }


def expand_group_action(group_action: np.ndarray, group_of_muscle: np.ndarray) -> np.ndarray:
    """把 138 维肌群动作展开为 700 维肌肉动作（等价于 `repeat_replace_x`）。"""
    group_action = np.asarray(group_action)
    out = group_action[..., group_of_muscle]
    return np.clip(out, -1.0, 1.0)


# ------------------------------------------------------------------ 路径 B


class DirectActor:
    """纯 torch 的 actor 前向实现（独立于 SB3）。"""

    def __init__(self, policy_pth: Path, muscle_groups: Sequence[Sequence[int]], device: str = "cpu"):
        import torch
        from torch import nn

        sd = torch.load(Path(policy_pth), map_location=device, weights_only=True)
        self._sd = sd
        self.device = device
        exp = build_group_expansion(muscle_groups)
        self.muscle_dims = exp["muscle_dims"]
        self.muscle_group_nums = exp["muscle_group_nums"]
        self.group_of_muscle = torch.as_tensor(exp["group_of_muscle"], device=device)

        def lin(prefix: str, out_name: str) -> "nn.Linear":
            w = sd[f"{prefix}.weight"]
            b = sd[f"{prefix}.bias"]
            layer = nn.Linear(w.shape[1], w.shape[0])
            layer.weight.data.copy_(w)
            layer.bias.data.copy_(b)
            return layer.to(device).eval()

        # 注意：SB3 的 `create_mlp(features_dim, -1, net_arch, activation_fn)` 在**每一层之后**
        # 都会加激活（包括最后一层），因此 latent_pi 以 ReLU 结尾。
        # 这一细节由 `scripts/verify_policy.py` 的双路径对比发现。
        self.latent_pi = nn.Sequential(
            lin("actor.latent_pi.0", "l0"), nn.ReLU(),
            lin("actor.latent_pi.2", "l2"), nn.ReLU(),
            lin("actor.latent_pi.4", "l4"), nn.ReLU(),
        )
        self.mu = lin("actor.mu", "mu")
        self.log_std = lin("actor.log_std", "logstd")

    def __call__(self, obs: np.ndarray, deterministic: bool = True, generator=None) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            h = self.latent_pi(x)
            mean = self.mu(h)
            if deterministic:
                act = torch.tanh(mean)
            else:
                log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
                eps = torch.randn(mean.shape, device=mean.device, generator=generator)
                act = torch.tanh(mean + torch.exp(log_std) * eps)
            out = act[..., self.group_of_muscle]
            out = torch.clamp(out, -1.0, 1.0)
        arr = out.cpu().numpy()
        return arr[0] if arr.shape[0] == 1 else arr


# ------------------------------------------------------------------ 路径 A


@dataclass
class PolicyStack:
    """官方策略栈：SB3 模型 + VecNormalize 观测归一化。"""

    model: Any
    vec_normalize: Any
    deterministic: bool = True
    load_notes: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """按官方 VecNormalize 做观测归一化；输入 (dim,) 或 (n_envs, dim)。"""
        arr = np.asarray(obs, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        out = self.vec_normalize.normalize_obs(arr)
        return out[0] if single else out

    def predict(self, normalized_obs: np.ndarray) -> np.ndarray:
        """给定**已归一化**观测，返回 [-1,1] 的肌群展开动作。"""
        arr = np.asarray(normalized_obs, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        action, _ = self.model.predict(arr, deterministic=self.deterministic)
        return action[0] if single else action


def load_sb3_stack(
    checkpoint_dir: Path,
    env,
    device: str = "cpu",
    deterministic: bool = True,
) -> PolicyStack:
    """加载官方 checkpoint + VecNormalize。

    Args:
        checkpoint_dir: 含 `checkpoint/best_model.zip` 与 `checkpoint/best_env.zip` 的目录。
        env: 单个（非向量化）环境，用于构造 `DummyVecEnv` 以载入 VecNormalize。
        device: torch 设备。
        deterministic: 是否用确定性动作（tanh(mean)）。
    """
    import json

    import stable_baselines3  # noqa: F401
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    install_numpy2_pickle_compat()
    paths.ensure_msgym_on_path()
    paths.ensure_dynsyn_scripts_on_path()
    paths.ensure_dynsyn_package_on_path()
    from DynSyn import SAC_DynSyn  # type: ignore

    ckpt = Path(checkpoint_dir) / "checkpoint"
    model_path = ckpt / "best_model.zip"
    env_path = ckpt / "best_env.zip"
    for p in (model_path, env_path):
        if not p.is_file():
            raise FileNotFoundError(f"缺少 checkpoint 文件: {p}")

    vec_normalize = load_vec_normalize(env_path, env)
    vec_env = vec_normalize.venv

    with patched_float_schedule() as records:
        model = SAC_DynSyn.load(str(model_path), device=device)
    model.policy.eval()
    stack = PolicyStack(
        model=model,
        vec_normalize=vec_normalize,
        deterministic=deterministic,
        device=device,
    )
    stack.load_notes = {
        "float_schedule_fallbacks": list(records),
        "vec_normalize_load_mode": getattr(vec_normalize, "_load_mode", "unknown"),
    }
    return stack


@contextlib.contextmanager
def patched_float_schedule(fallback_lr: float = 5e-4):
    """加载期为 ``FloatSchedule`` 提供安全回退。

    **背景**：checkpoint 由 Python 3.12 序列化，其 ``lr_schedule`` 是 cloudpickle 打包的
    **code object**。``COMPARE_OP`` 的编码在 Python 3.11 / 3.12 间不兼容，在 3.11 上执行会
    走错分支并对 ``warmup_fraction=0`` 做除法，因此**任何输入**都抛 ``ZeroDivisionError``
    （已实测：``fn(1.0)`` / ``fn(0.5)`` / ``fn(0.0)`` 全部失败）。SB3 在构建优化器时会调用
    ``lr_schedule(1)``，于是加载中断。

    **处理**：仅在加载期间把 ``FloatSchedule.__call__`` 包一层，异常时回退到线性学习率。
    评估阶段不做梯度更新，学习率取值不影响结果；策略网络权重是纯 torch tensor，
    不受此问题影响。加载完成后立刻恢复原方法。
    """
    from stable_baselines3.common.utils import FloatSchedule

    original = FloatSchedule.__call__
    records: list = []

    def safe_call(self, progress_remaining: float) -> float:
        try:
            return original(self, progress_remaining)
        except Exception as exc:  # 3.12 code object 在 3.11 上不可正确执行
            records.append(f"{type(exc).__name__}: {exc}")
            return float(fallback_lr) * max(0.0, float(progress_remaining))

    FloatSchedule.__call__ = safe_call  # type: ignore[method-assign]
    try:
        yield records
    finally:
        FloatSchedule.__call__ = original  # type: ignore[method-assign]


def load_vec_normalize(env_path: Path, env):
    """载入官方 ``best_env.zip``（观测归一化统计）。

    该文件是**裸 pickle**（不是 zip），且由 numpy 2.x 序列化。当前环境若为 numpy 1.x，
    直接用 ``VecNormalize.load`` 会因 ``numpy._core`` 与随机数构造函数签名差异而失败。
    这里先走官方路径，失败则回退到兼容 unpickler，随后按上游同样的方式绑定 env 与开关。
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_path = Path(env_path)
    if not env_path.is_file():
        raise FileNotFoundError(f"缺少 VecNormalize 文件: {env_path}")

    vec_env = DummyVecEnv([lambda: env])
    try:
        vec_normalize = VecNormalize.load(str(env_path), vec_env)
        mode = "official"
    except Exception as exc:  # numpy 2 pickle 在新/旧 numpy 间不兼容
        obj = compat_pickle_load(env_path)
        if obj is None:
            raise ValueError(f"VecNormalize 文件为空: {env_path}") from exc
        obj.set_venv(vec_env)
        obj.training = False
        obj.norm_reward = False
        vec_normalize = obj
        mode = f"compat_unpickler ({type(exc).__name__}: {exc})"
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    vec_normalize._load_mode = mode
    return vec_normalize


def load_policy_kwargs(checkpoint_dir: Path) -> Dict[str, Any]:
    """读取 checkpoint 的 `data`（policy_kwargs / dynsyn 分组等）。"""
    import json

    install_numpy2_pickle_compat()
    model_path = Path(checkpoint_dir) / "checkpoint" / "best_model.zip"
    with zipfile.ZipFile(model_path) as z:
        data = json.loads(z.read("data").decode())
    return data


def direct_actor_from_checkpoint(checkpoint_dir: Path, device: str = "cpu") -> DirectActor:
    """从 checkpoint 直接构建纯 torch actor。"""
    model_path = Path(checkpoint_dir) / "checkpoint" / "best_model.zip"
    kwargs = load_policy_kwargs(checkpoint_dir)
    groups = kwargs["policy_kwargs"]["dynsyn"]
    with zipfile.ZipFile(model_path) as z:
        payload = z.read("policy.pth")
    tmp = Path(checkpoint_dir) / "checkpoint" / "_policy_direct.pth"
    tmp.write_bytes(payload)
    return DirectActor(tmp, groups, device=device)


__all__ = [
    "PolicyStack",
    "DirectActor",
    "load_sb3_stack",
    "load_policy_kwargs",
    "direct_actor_from_checkpoint",
    "build_group_expansion",
    "expand_group_action",
    "LOG_STD_MIN",
    "LOG_STD_MAX",
]
