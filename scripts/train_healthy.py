"""正常行走微调入口：固定肌力 1.0，只改进长时稳定性。

## 设计要点（每一条都对应任务书里的要求）

**环境**：训练与评估都通过 :func:`hemirl.research_env.build_research_env`，
终止/计时/奖励语义与评估完全一致（不会把上游「偏离参考即终止 + 3.51 s 硬上限」带进训练）。

**算法**：沿用上游 ``DynSyn.SAC_DynSyn.SAC_DynSyn`` 与 ``Actor_DynSyn``，
加载官方 actor 权重与官方 ``VecNormalize`` 统计。不引入新算法、不重建模型、
不改观测与动作结构。

**各项参数的加载/初始化方式**（都会写进 ``run_meta.json``）：

================  ==========================================================
对象               处理方式
================  ==========================================================
actor             **从官方 checkpoint 加载**（含 dynsyn_layer 的 138 维输出头）
critic            由 ``--critic-init`` 决定：``keep`` 沿用官方 / ``reset`` 重新初始化
critic_target     与 critic 同步（``keep`` 时复制官方；``reset`` 时重新初始化）
温度参数 alpha    由 ``--alpha-init`` 决定：``keep`` 沿用 / ``reset`` 置 log_ent_coef=0
actor 优化器       重建（Adam，显式 lr）；不继承 checkpoint 的优化器状态
ent_coef 优化器     ``keep`` 时沿用 / ``reset`` 时重建
lr / 进度调度       **显式替换**为常数或线性，不继承已完成的旧进度
dynsyn_weight_amp  **显式固定为 0.0**（= 官方基线语义 weight≡1）
================  ==========================================================

**为什么要显式固定 dynsyn_weight_amp**：checkpoint 里它是 ``None``，
而 ``dynsyn_k=5e-9``、``dynsyn_a=3e7``、``num_timesteps=4.5e7``，
上游 ``train()`` 会算出 ``amp = min(5e-9·(4.5e7−3e7), 0.1) = 0.075``，
于是 ``weight = clamp(weight·0.1, −amp, amp) + 1``，
**静默地把肌群协同比改掉**。固定为 0.0 时 ``weight ≡ 1``，与官方基线一致
（该等价关系已由 ``reports/verify_policy.json`` 的 amp 语义检查证实）。

**预算**：``--max-transitions``（所有并行环境**合计**）与 ``--max-minutes`` 先到者停。
到预算即保存并提供续训命令，不会自动开启更长训练。

用法::

    # 2000 transition 的训练路径检查（含 critic/alpha 两种初始化的对照）
    MUJOCO_GL=egl PYTHONPATH=. python scripts/train_healthy.py --config configs/train_healthy_v1.json --check-only

    # 有限预算试训练
    MUJOCO_GL=egl PYTHONPATH=. python scripts/train_healthy.py --config configs/train_healthy_v1.json

    # 从上次结果续训
    MUJOCO_GL=egl PYTHONPATH=. python scripts/train_healthy.py --config configs/train_healthy_v1.json --resume
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402
from hemirl.research_env import ResearchEnvConfig, build_research_env  # noqa: E402
from hemirl.reward_healthy import HealthyRewardConfig  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402


# ------------------------------------------------------------------ 环境工厂


def make_research_env(spec: Dict[str, Any]):
    """顶层工厂：从**可 pickle 的 dict spec** 构建研究环境（供 SubprocVecEnv 使用）。"""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from hemirl import paths as _p
    from hemirl.research_env import RewardSplitConfig, ResearchEnvConfig
    from hemirl.research_env import build_research_env as _build
    from hemirl.reward_healthy import HealthyRewardConfig
    from hemirl.termination import TerminationConfig

    env_cfg = ResearchEnvConfig(
        termination=TerminationConfig(
            kind="research",
            min_pelvis_height=spec["min_pelvis_height"],
            max_pelvis_up_tilt_deg=spec["max_pelvis_up_tilt_deg"],
            time_limit_s=None,
        ),
        max_episode_seconds=spec["max_episode_seconds"],
        reward_mode="split",
        reward_split=RewardSplitConfig(
            w_survival=spec["w_survival"],
            include_official_healthy=spec["include_official_healthy"],
        ),
        extra_reward=HealthyRewardConfig(**spec["extra_reward"]) if spec.get("extra_reward") else None,
        keep_ledger=False,
        name=spec.get("name", "train"),
    )
    env, _raw = _build(env_cfg=env_cfg, checkpoint_dir=_p.checkpoint_dir("LocomotionFull"))
    return env


def env_spec(cfg: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, Any]:
    """把训练配置里的环境相关字段整理成可 pickle 的 spec。"""
    return {
        "max_episode_seconds": float(stage["max_episode_seconds"]),
        "min_pelvis_height": 0.55,
        "max_pelvis_up_tilt_deg": 60.0,
        "w_survival": float(cfg["reward_split"]["w_survival"]),
        "include_official_healthy": bool(cfg["reward_split"]["include_official_healthy"]),
        "extra_reward": cfg.get("extra_reward"),
        "name": stage.get("name", "train"),
    }


# ------------------------------------------------------------------ 模型装配


def load_official_model(vec_env, cfg: Dict[str, Any], ckpt_dir: Path):
    """加载官方 DynSyn-SAC，并按配置处理 critic / alpha / lr / amp。返回 (model, notes)。"""
    import torch
    from stable_baselines3.common.buffers import ReplayBuffer

    from hemirl import policy as pol

    install = pol.install_numpy2_pickle_compat()
    paths.ensure_msgym_on_path()
    paths.ensure_dynsyn_scripts_on_path()
    paths.ensure_dynsyn_package_on_path()
    from DynSyn import SAC_DynSyn  # type: ignore

    model_path = ckpt_dir / "checkpoint" / "best_model.zip"
    notes: Dict[str, Any] = {
        "load": {"checkpoint": str(model_path), "sha256": provenance.file_sha256(model_path)},
    }

    with pol.patched_float_schedule() as records:
        model = SAC_DynSyn.load(str(model_path), env=vec_env, device="cpu", print_system_info=False)
    notes["load"]["float_schedule_fallbacks"] = list(records)
    notes["inherited"] = {
        "num_timesteps": int(model.num_timesteps),
        "_total_timesteps": float(getattr(model, "_total_timesteps", float("nan"))),
        "dynsyn_k": float(model.dynsyn_k),
        "dynsyn_a": float(model.dynsyn_a),
        "dynsyn_weight_amp": model.dynsyn_weight_amp,
        "buffer_size": int(model.buffer_size),
        "learning_starts": float(model.learning_starts),
        "batch_size": int(model.batch_size),
        "train_freq": str(model.train_freq),
        "gradient_steps": int(model.gradient_steps),
        "ent_coef_value": float(torch.exp(model.log_ent_coef).item())
        if model.log_ent_coef is not None
        else None,
        "target_entropy": float(model.target_entropy),
        "actor_optimizer_lr": float(model.actor.optimizer.param_groups[0]["lr"]),
    }
    try:
        notes["inherited"]["lr_schedule_at_1p0"] = float(model.lr_schedule(1.0))
        notes["inherited"]["lr_schedule_at_0p0"] = float(model.lr_schedule(0.0))
    except Exception as exc:  # 3.11 上 code object 不兼容
        notes["inherited"]["lr_schedule_error"] = f"{type(exc).__name__}: {exc}"

    # --- actor / critic / target 的处理
    critic_init = cfg.get("critic_init", "keep")
    n_reset_qf = 0
    if critic_init == "reset":
        # 注意两个坑：
        #  (a) SB3 的 ``ContinuousCritic`` **没有** ``reset_parameters()``；
        #  (b) 不能对整个 critic 递归重置 —— ``features_extractor`` 与 actor **共享**，
        #      递归会把 actor 的特征提取器也一起重置掉。
        # 因此只重置 critic 自己的 ``q_networks`` 里的 Linear 层。
        for qf in model.critic.q_networks:
            for mod in qf.modules():
                if isinstance(mod, torch.nn.Linear):
                    mod.reset_parameters()
                    n_reset_qf += 1
        model.critic_target.load_state_dict(model.critic.state_dict())
    notes["critic_init"] = {
        "mode": critic_init,
        "actor_loaded_from_checkpoint": True,
        "critic_loaded_from_checkpoint": critic_init == "keep",
        "critic_target_synced": True,
        "n_linear_layers_reset": n_reset_qf,
        "feature_extractor_shared_with_actor": True,
        "feature_extractor_reset": False,
    }

    # --- 温度参数
    alpha_init = cfg.get("alpha_init", "keep")
    if alpha_init == "reset":
        with torch.no_grad():
            model.log_ent_coef.fill_(0.0)
        params = [model.log_ent_coef]
        # 注意：checkpoint 的 ``learning_rate`` 是 cloudpickle 的**函数**，不是 float
        model.ent_coef_optimizer = torch.optim.Adam(params, lr=float(cfg["learning_rate"]))
    notes["alpha_init"] = {
        "mode": alpha_init,
        "ent_coef_after": float(torch.exp(model.log_ent_coef).item()),
        "target_entropy": float(model.target_entropy),
    }

    # --- DynSyn 协同比：必须显式固定
    amp = cfg.get("dynsyn_weight_amp", 0.0)
    model.dynsyn_weight_amp = float(amp)
    model.actor.dynsyn_layer.update_dynsyn_weight_amp(float(amp))
    would_be = float(model.get_dynsyn_weight_amp(model.dynsyn_k, model.dynsyn_a, model.num_timesteps))
    notes["dynsyn_weight_amp"] = {
        "pinned_to": float(amp),
        "schedule_would_have_given": would_be,
        "schedule_avoided": bool(abs(would_be - float(amp)) > 1e-12),
        "meaning": "amp=0 → weight≡1，与官方基线动作语义一致",
        "layer_value": model.actor.dynsyn_layer.dynsyn_weight_amp,
    }

    # --- 学习率与进度调度：显式替换
    lr = float(cfg["learning_rate"])
    scheme = cfg.get("lr_schedule", "constant")
    if scheme == "constant":
        sched = lambda progress_remaining: lr  # noqa: E731
    elif scheme == "linear":
        sched = lambda progress_remaining: lr * max(0.0, float(progress_remaining))  # noqa: E731
    else:
        raise ValueError(f"未知 lr_schedule: {scheme}")
    model.lr_schedule = sched
    for opt in (model.actor.optimizer, model.critic.optimizer):
        for g in opt.param_groups:
            g["lr"] = lr
    notes["learning_rate"] = {
        "scheme": scheme,
        "value": lr,
        "actor_optimizer_lr_after": float(model.actor.optimizer.param_groups[0]["lr"]),
        "critic_optimizer_lr_after": float(model.critic.optimizer.param_groups[0]["lr"]),
        "optimizer_state": "actor/critic 优化器状态**重建**（不继承已完成的旧进度）",
    }

    # --- buffer / 节奏 / 日志
    model.buffer_size = int(cfg["replay_buffer_size"])
    model.batch_size = int(cfg["batch_size"])
    model.learning_starts = int(cfg["learning_starts"])
    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

    model.train_freq = TrainFreq(int(cfg["train_freq"]), TrainFrequencyUnit.STEP)
    model.gradient_steps = int(cfg["gradient_steps"])
    # ``SAC.load`` 已按 checkpoint 的 buffer_size=1e6 建过一个 buffer（约 29.4 GiB）；
    # 而 ``_setup_model`` 只在 load/__init__ 时调用，``learn`` 不会重建，
    # 所以必须**自己按新容量重建**，不能只置 None。
    buf_cls = type(model.replay_buffer) if model.replay_buffer is not None else ReplayBuffer
    model.replay_buffer = buf_cls(
        model.buffer_size,
        model.observation_space,
        model.action_space,
        device=model.device,
        n_envs=model.n_envs,
        optimize_memory_usage=model.optimize_memory_usage,
        **model.replay_buffer_kwargs,
    )
    # checkpoint 里 tensorboard_log 指向训练机的相对路径，且本环境未装 tensorboard；
    # 置 None 并显式记录，避免把日志悄悄写到别处（或直接报错）。
    model.tensorboard_log = None
    model.verbose = 0
    model.set_random_seed(int(cfg.get("seed", 0)))
    notes["buffer"] = {
        "size": model.buffer_size,
        "bytes_per_transition": 4 * 3601 * 2 + 4 * 700 + 4 + 1,
        "est_gib": (4 * 3601 * 2 + 4 * 700 + 4 + 1) * model.buffer_size / 1024**3,
    }
    # **重要**：SB3 的 ``OffPolicyAlgorithm.learn`` 是「每次 collect_rollouts 之后调用一次
    # ``train()``」，而一次 collect_rollouts 收集 ``train_freq.frequency × n_envs`` 个 transition。
    # 所以更新次数 = transitions / (train_freq × n_envs)，**不是** transitions / train_freq。
    # 早期按后者估算会让实际更新次数少 n_envs 倍。
    notes["train_freq"] = {
        "frequency": int(cfg["train_freq"]),
        "unit": "step（每个子环境每轮收集的步数）",
        "gradient_steps": model.gradient_steps,
        "batch_size": model.batch_size,
        "learning_starts": int(model.learning_starts),
        "n_envs": int(model.n_envs),
        "transitions_per_update": int(cfg["train_freq"]) * int(model.n_envs),
        "formula": "n_updates = total_transitions / (train_freq × n_envs)",
    }
    return model, notes


def save_checkpoint(model, vec_normalize, run_dir: Path, cfg: Dict[str, Any], meta: Dict[str, Any],
                    tag_name: str = "best") -> Dict[str, str]:
    """按官方 checkpoint 的目录布局保存，使同一套加载代码可直接评估。"""
    ckpt = run_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt / f"{tag_name}_model"))
    vec_normalize.save(str(ckpt / f"{tag_name}_env.zip"))
    provenance.write_json(run_dir / "train_meta.json", meta)
    if not (run_dir / "locomotionFull.json").is_file():
        src = Path(cfg["checkpoint_dir"]) / "locomotionFull.json"
        if src.is_file():
            shutil.copy2(src, run_dir / "locomotionFull.json")
    return {
        "model": str(ckpt / f"{tag_name}_model.zip"),
        "vec_normalize": str(ckpt / f"{tag_name}_env.zip"),
    }


# ------------------------------------------------------------------ 评估


def quick_eval(model, vec_normalize, spec: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    """在给定种子上跑一次确定性评估（单进程，不更新归一化统计）。"""
    from hemirl.research_env import ResearchEnvConfig as RC
    from hemirl.termination import TerminationConfig as TC

    env_cfg = RC(
        termination=TC(kind="research", min_pelvis_height=spec["min_pelvis_height"],
                       max_pelvis_up_tilt_deg=spec["max_pelvis_up_tilt_deg"]),
        max_episode_seconds=spec["max_episode_seconds"],
        reward_mode="split",
        keep_ledger=False,
        name="val",
    )
    env, raw = build_research_env(env_cfg=env_cfg, checkpoint_dir=paths.checkpoint_dir("LocomotionFull"))
    vn = vec_normalize
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        alive = 0.0
        steps = 0
        while True:
            z = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(z, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            steps += 1
            alive = info["termination"]["elapsed_time_s"]
            if terminated or truncated:
                out.append({
                    "seed": seed,
                    "steps": steps,
                    "alive_time_s": alive,
                    "termination_source": info["termination"]["termination_source"],
                    "completed": info["termination"]["termination_source"] in ("time_limit",),
                })
                break
    env.close()
    n = len(out)
    return {
        "n": n,
        "success_rate": float(sum(1 for r in out if r["completed"]) / n) if n else 0.0,
        "mean_alive_time_s": float(np.mean([r["alive_time_s"] for r in out])) if n else 0.0,
        "episodes": out,
        "seeds": seeds,
    }


# ------------------------------------------------------------------ 主流程


def main() -> int:
    parser = argparse.ArgumentParser(description="正常行走微调")
    parser.add_argument("--config", type=str, default=str(paths.CONFIGS_ROOT / "train_healthy_v1.json"))
    parser.add_argument("--tag", type=str, default=None, help="覆盖配置里的 tag")
    parser.add_argument("--out-root", type=str, default=str(paths.RUNS_ROOT / "train"))
    parser.add_argument("--resume", action="store_true", help="从 <run_dir>/checkpoint/last_model.zip 继续")
    parser.add_argument("--check-only", action="store_true", help="只做 2000-transition 训练路径检查")
    parser.add_argument("--check-transitions", type=int, default=2000)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--critic-init", choices=["keep", "reset"], default=None)
    parser.add_argument("--alpha-init", choices=["keep", "reset"], default=None)
    parser.add_argument("--no-extra-reward", action="store_true")
    parser.add_argument("--parallel-envs", type=int, default=None, help="覆盖并行环境数")
    parser.add_argument("--replay-buffer-size", type=int, default=None, help="覆盖 replay buffer 容量")
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.tag:
        cfg["tag"] = args.tag
    if args.critic_init:
        cfg["critic_init"] = args.critic_init
    if args.alpha_init:
        cfg["alpha_init"] = args.alpha_init
    if args.no_extra_reward:
        cfg["extra_reward"] = {"enable": False}
    if args.parallel_envs is not None:
        cfg["parallel_envs"] = int(args.parallel_envs)
    if args.replay_buffer_size is not None:
        cfg["replay_buffer_size"] = int(args.replay_buffer_size)
    if args.learning_rate is not None:
        cfg["learning_rate"] = float(args.learning_rate)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    run_dir = Path(args.out_root) / cfg["tag"]
    run_dir.mkdir(parents=True, exist_ok=True)

    budget_transitions = args.max_transitions or cfg["budget"]["max_transitions"]
    budget_minutes = args.max_minutes or cfg["budget"]["max_minutes"]
    if args.check_only:
        budget_transitions = args.check_transitions
        budget_minutes = 10.0

    stages = cfg["curriculum"]["stages"]
    stage = stages[0]
    spec_stage = env_spec(cfg, stage)

    print(f"=== 正常行走微调：{cfg['tag']} ===")
    print(f"  预算: {budget_transitions} transitions（所有并行环境合计） / {budget_minutes} 分钟，先到者停")
    print(f"  并行环境: {cfg['parallel_envs']}  buffer={cfg['replay_buffer_size']}  train_freq={cfg['train_freq']}")
    print(f"  课程起点: {stage['name']} = {stage['max_episode_seconds']} s")

    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    n_envs = int(cfg["parallel_envs"])
    vec_env = SubprocVecEnv([functools.partial(make_research_env, spec_stage) for _ in range(n_envs)])

    # --- 归一化：加载官方统计；训练与评估都冻结
    vn_src = ckpt_dir / "checkpoint" / "best_env.zip"
    try:
        vec_normalize = VecNormalize.load(str(vn_src), vec_env)
        vn_mode = "official"
    except Exception as exc:
        from hemirl.numpy_compat import compat_pickle_load

        obj = compat_pickle_load(vn_src)
        obj.set_venv(vec_env)
        vec_normalize = obj
        vn_mode = f"compat ({type(exc).__name__})"
    update_stats = bool(cfg.get("vec_normalize_update", False))
    vec_normalize.training = update_stats
    vec_normalize.norm_reward = False
    vec_normalize.clip_obs = float(cfg.get("clip_obs", 10.0))
    print(f"  VecNormalize: {vn_mode}, training(update)={vec_normalize.training}, clip_obs={vec_normalize.clip_obs}")

    # --- 模型
    model, notes = load_official_model(vec_normalize, cfg, ckpt_dir)
    if args.resume:
        last = run_dir / "checkpoint" / "last_model.zip"
        if not last.is_file():
            print(f"错误：--resume 需要 {last}")
            return 2
        model = type(model).load(str(last), env=vec_normalize, device="cpu", print_system_info=False)
        # 恢复后重新应用本项目的显式设置（load 会把 lr_schedule / 优化器 / amp 换回 checkpoint 的值）
        model, notes2 = load_official_model(vec_normalize, cfg, ckpt_dir)
        _reloaded = type(model).load(str(last), env=vec_normalize, device="cpu", print_system_info=False)
        model = _reloaded
        model.dynsyn_weight_amp = float(cfg.get("dynsyn_weight_amp", 0.0))
        model.actor.dynsyn_layer.update_dynsyn_weight_amp(float(cfg.get("dynsyn_weight_amp", 0.0)))
        _lr = float(cfg["learning_rate"])
        model.lr_schedule = (lambda progress_remaining, _lr=_lr: _lr)
        for _opt in (model.actor.optimizer, model.critic.optimizer):
            for _g in _opt.param_groups:
                _g["lr"] = _lr
        notes["resumed_from"] = str(last)
        print(f"  已从 {last} 恢复")

    meta: Dict[str, Any] = {
        "entry": "scripts/train_healthy.py",
        "config": cfg,
        "config_path": str(args.config),
        "notes": notes,
        "budget": {"transitions": budget_transitions, "minutes": budget_minutes},
        "check_only": bool(args.check_only),
        "resumed": bool(args.resume),
    }

    # --- 固定探针：衡量「策略漂移」用（同一批观测上动作的变化）
    probe_env, _probe_raw = build_research_env(
        env_cfg=ResearchEnvConfig(
            termination=TerminationConfig(kind="research"),
            max_episode_seconds=stage["max_episode_seconds"],
            keep_ledger=False,
            name="probe",
        ),
        checkpoint_dir=ckpt_dir,
    )
    probe_obs: List[np.ndarray] = []
    obs, _ = probe_env.reset(seed=0)
    for _ in range(40):
        probe_obs.append(np.asarray(vec_normalize.normalize_obs(np.asarray(obs, dtype=np.float32)),
                                    dtype=np.float32))
        a, _ = model.predict(probe_obs[-1], deterministic=True)
        obs, *_ = probe_env.step(a)
    probe_before = np.asarray([model.predict(o, deterministic=True)[0] for o in probe_obs])

    def policy_drift() -> Dict[str, float]:
        now = np.asarray([model.predict(o, deterministic=True)[0] for o in probe_obs])
        d = np.abs(now - probe_before)
        return {"max_abs_action_diff": float(d.max()), "mean_abs_action_diff": float(d.mean())}

    # --- 训练（用 callback 控制预算与课程）
    from stable_baselines3.common.callbacks import BaseCallback

    class BudgetCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t0 = time.perf_counter()
            self.start_transitions = int(model.num_timesteps)
            self.stop_reason: Optional[str] = None
            self.history: List[Dict[str, Any]] = []
            self.meta = meta

        def _on_step(self) -> bool:
            elapsed_min = (time.perf_counter() - self.t0) / 60.0
            done = self.num_timesteps - self.start_transitions
            if done >= budget_transitions:
                self.stop_reason = "到达 transition 预算"
                return False
            if elapsed_min >= budget_minutes:
                self.stop_reason = "到达墙钟预算"
                return False
            return True

    cb = BudgetCallback()
    n_updates_before = int(getattr(model, "_n_updates", 0))
    t0 = time.perf_counter()
    print(f"\n开始训练…（上限 {budget_transitions} transitions / {budget_minutes} 分钟）")
    model.learn(
        total_timesteps=int(budget_transitions) + 1000,  # 上限由 callback 控制
        callback=cb,
        reset_num_timesteps=False,
        log_interval=100,
        progress_bar=False,
    )
    wall_min = (time.perf_counter() - t0) / 60.0
    done_transitions = int(model.num_timesteps - cb.start_transitions)
    n_updates_done = int(getattr(model, "_n_updates", 0)) - n_updates_before

    drift = policy_drift()
    print(f"\n训练结束：{cb.stop_reason}")
    print(f"  实际 transitions = {done_transitions}  墙钟 = {wall_min:.2f} 分钟  "
          f"({done_transitions / max(1e-9, wall_min * 60):.1f} transitions/s)")
    print(f"  梯度更新次数 = {n_updates_done}  ({n_updates_done / max(1e-9, wall_min*60):.2f} updates/s)")
    print(f"  策略漂移（固定 40 个探针观测）: max|Δa|={drift['max_abs_action_diff']:.5f} "
          f"mean|Δa|={drift['mean_abs_action_diff']:.5f}")

    # 数值健全性
    finite_ok = all(torch.isfinite(p).all().item() for p in model.actor.parameters())
    critic_finite = all(torch.isfinite(p).all().item() for p in model.critic.parameters())
    alpha_val = float(torch.exp(model.log_ent_coef).item())
    print(f"  数值有限性: actor={finite_ok} critic={critic_finite} alpha={alpha_val:.4f}")
    print(f"  dynsyn_weight_amp（训练后）= {model.actor.dynsyn_layer.dynsyn_weight_amp}（应恒为 0.0）")

    # --- 验证集评估
    val_spec = dict(spec_stage)
    val_res = quick_eval(model, vec_normalize, val_spec, list(cfg["curriculum"]["val_seeds"]))
    print(f"  验证集（{val_res['seeds']}）: 成功率={val_res['success_rate']:.2f} "
          f"平均存活={val_res['mean_alive_time_s']:.2f}s")

    n_this_run = int(done_transitions)
    n_total = int(model.num_timesteps - notes["inherited"]["num_timesteps"]) + int(
        notes["inherited"]["num_timesteps"] - notes["inherited"]["num_timesteps"]
    )
    n_total = int(model.num_timesteps)
    meta.update({
        "result": {
            # ``transitions`` 保留为**本次运行**的 transition 数（向后兼容）；
            # 另给 ``transitions_total`` = 该模型累计训练过的环境步数（含 checkpoint 自带的 4.5e7）。
            "transitions": n_this_run,
            "transitions_this_run": n_this_run,
            "transitions_total_since_official": int(model.num_timesteps),
            "transitions_per_s": done_transitions / max(1e-9, wall_min * 60),
            "n_gradient_updates": n_updates_done,
            "updates_per_s": n_updates_done / max(1e-9, wall_min * 60),
            "stop_reason": cb.stop_reason,
            "policy_drift": drift,
            "actor_params_finite": bool(finite_ok),
            "critic_params_finite": bool(critic_finite),
            "ent_coef_after": alpha_val,
            "dynsyn_weight_amp_after": model.actor.dynsyn_layer.dynsyn_weight_amp,
            "val": val_res,
        },
        "notes": notes,
    })

    # --- 保存
    saved = save_checkpoint(model, vec_normalize, run_dir, cfg, meta, "last")
    saved_best = save_checkpoint(model, vec_normalize, run_dir, cfg, meta, "best")
    print(f"  已保存: {saved['model']}")

    # 重载校验
    reload_ok = False
    try:
        m2 = type(model).load(saved["model"], env=vec_normalize, device="cpu", print_system_info=False)
        z = vec_normalize.normalize_obs(probe_obs[0])
        a1 = model.predict(z, deterministic=True)[0]
        a2 = m2.predict(z, deterministic=True)[0]
        reload_ok = bool(np.allclose(a1, a2, atol=1e-6))
        meta["result"]["reload_action_match"] = reload_ok
    except Exception as exc:
        meta["result"]["reload_error"] = f"{type(exc).__name__}: {exc}"
    print(f"  保存/重载一致性: {reload_ok}")

    meta["saved"] = saved
    meta["provenance"] = provenance.run_provenance(
        entry="scripts/train_healthy.py",
        args=vars(args),
        extra={"train_config": cfg, "notes": notes},
        checkpoint_dir=ckpt_dir,
    )
    provenance.write_json(run_dir / "train_meta.json", meta)
    provenance.write_json(run_dir / "run.json", meta["provenance"])

    # append-only 训练历史：续训会覆盖 train_meta.json，因此需要一份不丢记录的账本
    hist_path = run_dir / "train_history.json"
    hist = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.is_file() else {"entries": []}
    hist["entries"].append({
        "recorded_at_utc": provenance.build_provenance()["generated_at_utc"],
        "resumed": bool(args.resume),
        "resumed_from": notes.get("resumed_from"),
        "transitions_this_run": n_this_run,
        "num_timesteps_after": int(model.num_timesteps),
        "wall_minutes": wall_min,
        "stop_reason": cb.stop_reason,
        "critic_init": cfg.get("critic_init"),
        "alpha_init": cfg.get("alpha_init"),
        "dynsyn_weight_amp_pinned": float(cfg.get("dynsyn_weight_amp", 0.0)),
        "policy_drift": drift,
        "val_success_rate": val_res["success_rate"],
        "val_mean_alive_time_s": val_res["mean_alive_time_s"],
        "saved_model_sha256": provenance.file_sha256(Path(saved["model"])),
    })
    hist["total_transitions_this_tag"] = int(sum(e["transitions_this_run"] for e in hist["entries"]))
    provenance.write_json(hist_path, hist)
    print(f"  历史已追加到 {hist_path}（本 tag 累计 {hist['total_transitions_this_tag']} transitions）")

    probe_env.close()
    vec_env.close()

    print(f"\n输出目录: {run_dir}")
    print("续训命令:")
    print(f"  MUJOCO_GL=egl PYTHONPATH=. python scripts/train_healthy.py "
          f"--config {args.config} --resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
