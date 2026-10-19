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

# CurriculumCallback 定义在模块级（便于单测直接实例化），因此这里必须在导入期就
# 拿到 BaseCallback。SB3 本就是本脚本的硬依赖。
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

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
            max_lateral_dev_from_ref_m=spec.get("max_lateral_dev_from_ref_m"),
            lateral_hold_s=float(spec.get("lateral_hold_s") or 0.0),
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
        "max_lateral_dev_from_ref_m": cfg.get("termination", {}).get(
            "max_lateral_dev_from_ref_m"
        ),
        "lateral_hold_s": cfg.get("termination", {}).get("lateral_hold_s", 0.0),
        "w_survival": float(cfg["reward_split"]["w_survival"]),
        "include_official_healthy": bool(cfg["reward_split"]["include_official_healthy"]),
        "extra_reward": cfg.get("extra_reward"),
        # 验证/升级判定必须使用与最终评估同一套质量阈值，否则「达标」是另一个标准
        "success": cfg.get("eval", {}).get("success"),
        "name": stage.get("name", "train"),
    }


# ------------------------------------------------------------------ 课程


class CurriculumCallback(BaseCallback):
    """真正的课程阶段切换：**由独立验证集结果触发**，不是由训练步数触发。

    设计要点（每条都对应一个曾经真实出错或明确禁止的做法）:

    1. **升级必须由验证集触发**。`min_transitions` 只是「最早可以评估」的门槛，
       单独达到步数绝不升级。
    2. **验证用与训练完全相同的 spec**（同一终止判据、同一回合时长），
       且与最终测试种子严格分离。
    3. **不通过就一直留在当前阶段**，不会为了「晋级」而降低门槛；
       停滞会在 `stall_after_transitions` 后记录一次 `stalled` 事件供人工分析。
    4. **切换时长不重建环境进程**：通过 ``vec_env.env_method`` 广播到所有 worker。
       重建 ``SubprocVecEnv`` 会丢失 worker 内部状态、重新加载 MuJoCo 模型
       （实测约 2 s × n_envs），并使 ``VecNormalize`` 的挂载关系需要重新建立。
    5. 每次评估/升级都落盘到 ``curriculum_history.json``，可事后核查。
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        stages: List[Dict[str, Any]],
        spec_builder,
        budget_transitions: int,
        budget_minutes: float,
        vec_normalize=None,
        model=None,
        resume_state: Optional[Dict[str, Any]] = None,
        run_dir: Optional[Path] = None,
        checkpoint_every: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.stages = list(stages)
        self.spec_builder = spec_builder
        self.budget_transitions = int(budget_transitions)
        self.budget_minutes = float(budget_minutes)
        self._vec_normalize = vec_normalize
        self.cur = cfg["curriculum"]
        self.val_seeds = list(self.cur["val_seeds"])
        self.val_every = int(self.cur.get("val_every_transitions", 5000))
        self.promote_at = float(self.cur["promote_val_success_rate"])
        self.stall_after = int(self.cur.get("stall_after_transitions", 25000))
        self.run_dir = run_dir
        self.checkpoint_every = int(checkpoint_every)
        self._next_ckpt_at = int(checkpoint_every) if checkpoint_every > 0 else 10**12
        self._model_ref = None  # _on_training_start 里填

        self.t0 = time.perf_counter()
        self.start_transitions = 0  # 在 _on_training_start 里填
        self.stage_idx = 0
        self.transitions_in_stage = 0
        self.stop_reason: Optional[str] = None
        self.history: List[Dict[str, Any]] = []
        self.last_val: Optional[Dict[str, Any]] = None
        self._next_val_at = 0
        self._stall_logged = False

        # 续训恢复：课程阶段/阶段内计数/事件历史。文件里存的是什么就恢复什么，
        # 恢复后**仍向所有 worker 广播当前阶段**（worker 是新建的，默认在 stages[0]）。
        if resume_state:
            idx = int(resume_state.get("stage_idx", 0))
            if not (0 <= idx < len(self.stages)):
                raise ValueError(f"课程状态里的 stage_idx={idx} 越界（共 {len(self.stages)} 阶段）")
            self.stage_idx = idx
            self.transitions_in_stage = int(resume_state.get("transitions_in_stage", 0))
            self.history = list(resume_state.get("history", []))
            self._next_val_at = int(self._current_stage().get("min_transitions", 0))
            self._stall_logged = bool(resume_state.get("stop_reason") is None
                                      and any(e.get("event") == "stalled" for e in self.history))
            self.history.append({
                "event": "resume",
                "stage": self._current_stage()["name"],
                "stage_seconds": float(self._current_stage()["max_episode_seconds"]),
                "transitions_in_stage": self.transitions_in_stage,
                "note": "从 curriculum_state 恢复；worker 在 _on_training_start 广播该阶段时长",
            })
            self._write_history()

    # ---------------------------------------------------------------- 工具

    def _elapsed_min(self) -> float:
        return (time.perf_counter() - self.t0) / 60.0

    def _done(self) -> int:
        return int(self.num_timesteps) - self.start_transitions

    def _current_stage(self) -> Dict[str, Any]:
        return self.stages[self.stage_idx]

    def _write_history(self) -> None:
        run_dir = Path(self.cfg.get("_run_dir", "."))
        try:
            provenance.write_json(run_dir / "curriculum_history.json", {
                "stages_config": self.stages,
                "val_seeds": self.val_seeds,
                "promote_val_success_rate": self.promote_at,
                "events": self.history,
            })
        except Exception:  # noqa: BLE001 - 落盘失败不应中断训练
            pass

    # ---------------------------------------------------------------- 回调

    def _on_training_start(self) -> None:
        self.start_transitions = int(self.model.num_timesteps)
        self._model_ref = self.model
        self._next_val_at = max(int(self._current_stage().get("min_transitions", 0)),
                                int(self.transitions_in_stage))
        # worker 环境是新建的，默认在 stages[0] 的时长；把当前阶段广播到全部 worker
        # （全新训练时这也是无害的自洽操作）
        secs = float(self._current_stage()["max_episode_seconds"])
        try:
            applied = self.training_env.env_method("configure_episode_seconds", secs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"训练开始时广播阶段时长失败（{secs}s）：{exc}") from exc
        self.history.append({
            "event": "start",
            "stage": self._current_stage()["name"],
            "max_episode_seconds": secs,
            "num_timesteps": self.start_transitions,
            "broadcast_to_workers": len(applied) if applied else 0,
            "worker_reply": applied[0] if applied else None,
        })
        self._write_history()

    def _evaluate_stage(self) -> Dict[str, Any]:
        spec = self.spec_builder(self.cfg, self._current_stage())
        res = quick_eval(self.model, self._vn, spec, self.val_seeds)
        # 存活探针：按固定时长（如 10 s）额外跑一次，记录「10 s 存活率」与跌倒时刻，
        # 与当前阶段无关（阶段 C 要求逐次验证都记录）。
        probe = None
        probe_s = float(self.cur.get("survival_probe_seconds", 0) or 0)
        if probe_s > 0 and abs(probe_s - spec["max_episode_seconds"]) > 1e-9:
            probe_spec = dict(spec)
            probe_spec["max_episode_seconds"] = probe_s
            probe = quick_eval(self.model, self._vn, probe_spec, self.val_seeds)
        self.last_val = res
        rec = {
            "event": "validate",
            "stage": self._current_stage()["name"],
            "stage_seconds": spec["max_episode_seconds"],
            "transitions_in_stage": int(self.transitions_in_stage),
            "num_timesteps": int(self.num_timesteps),
            "success_rate": res["success_rate"],
            "complete_rate": res["complete_rate"],
            "mean_alive_time_s": res["mean_alive_time_s"],
            "end_sources": _count_sources(res["episodes"]),
            "elapsed_min": self._elapsed_min(),
        }
        if probe is not None:
            rec["survival_probe"] = {
                "seconds": probe_s,
                "complete_rate": probe["complete_rate"],
                "mean_alive_time_s": probe["mean_alive_time_s"],
                "fall_times_s": [round(e["alive_time_s"], 2) for e in probe["episodes"]],
                "mean_forward_speed": float(np.mean(
                    [e["mean_forward_speed"] for e in probe["episodes"] if e.get("mean_forward_speed") is not None]
                )) if any(e.get("mean_forward_speed") is not None for e in probe["episodes"]) else None,
                "end_sources": _count_sources(probe["episodes"]),
            }
        self.history.append(rec)
        self._write_history()
        print(f"\n[课程] 阶段 {rec['stage']} ({rec['stage_seconds']:.0f}s) 验证: "
              f"成功率={res['success_rate']:.2f} 走满率={res['complete_rate']:.2f} "
              f"平均存活={res['mean_alive_time_s']:.2f}s 结束原因={rec['end_sources']}")
        if probe is not None:
            print(f"[课程] 10 s 存活探针: 走满率={probe['complete_rate']:.2f} "
                  f"平均存活={probe['mean_alive_time_s']:.2f}s "
                  f"跌倒时刻={rec['survival_probe']['fall_times_s']} "
                  f"速度={rec['survival_probe']['mean_forward_speed']}")
        return res

    def _promote(self, val: Dict[str, Any]) -> None:
        old = self._current_stage()
        self.stage_idx += 1
        new = self._current_stage()
        secs = float(new["max_episode_seconds"])
        # 广播到全部 worker（不重建进程）
        try:
            applied = self.training_env.env_method("configure_episode_seconds", secs)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"课程切换广播失败（阶段 {new['name']}）：{exc}") from exc
        self.transitions_in_stage = 0
        self._stall_logged = False
        self._next_val_at = int(new.get("min_transitions", 0))
        self.history.append({
            "event": "promote",
            "from_stage": old["name"],
            "to_stage": new["name"],
            "to_seconds": secs,
            "val_success_rate": val["success_rate"],
            "num_timesteps": int(self.num_timesteps),
            "applied_in_workers": applied[0] if applied else None,
            "n_workers": len(applied) if applied else 0,
            "elapsed_min": self._elapsed_min(),
        })
        self._write_history()
        print(f"[课程] 升级 {old['name']} → {new['name']}（{secs:.0f} s）；"
              f"已在 {len(applied) if applied else 0} 个 worker 生效: "
              f"{applied[0] if applied else None}")

    def _on_step(self) -> bool:
        self.transitions_in_stage += self.training_env.num_envs

        # 周期性完整检查点（模型+归一化+经验池+课程状态），供断点续训
        if (self.checkpoint_every > 0 and self._model_ref is not None
                and self._done() >= self._next_ckpt_at):
            n = self._done()
            ckpt_dir = self.run_dir / "checkpoints" / f"ckpt_{n}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            saved = save_checkpoint(self._model_ref, self._vn, self.run_dir, self.cfg, {},
                                    tag_name="ckpt", cb=self, save_buffer=True, sub_dir=ckpt_dir)
            self.history.append({
                "event": "checkpoint",
                "transitions_done": n,
                "stage": self._current_stage()["name"],
                "files": {k: str(Path(v).name) for k, v in saved.items() if k != "curriculum_state"},
                "replay_buffer_size_bytes": int(Path(saved["replay_buffer"]).stat().st_size)
                if "replay_buffer" in saved else None,
                "elapsed_min": self._elapsed_min(),
            })
            self._write_history()
            self._next_ckpt_at = n + self.checkpoint_every
            print(f"[检查点] {n} transitions：{saved}")

        if self._done() >= self.budget_transitions:
            self.stop_reason = "到达 transition 预算"
            return False
        if self._elapsed_min() >= self.budget_minutes:
            self.stop_reason = "到达墙钟预算"
            return False

        stage = self._current_stage()
        min_t = int(stage.get("min_transitions", 0))
        at_last = self.stage_idx >= len(self.stages) - 1
        if at_last and self._next_val_at > 10**12:
            return True
        if self.transitions_in_stage >= min_t and self.transitions_in_stage >= self._next_val_at:
            val = self._evaluate_stage()
            self._next_val_at = self.transitions_in_stage + self.val_every
            if at_last:
                return True
            if val["success_rate"] >= self.promote_at:
                self._promote(val)
            elif self.transitions_in_stage >= self.stall_after and not self._stall_logged:
                self._stall_logged = True
                self.history.append({
                    "event": "stalled",
                    "stage": stage["name"],
                    "stage_seconds": float(stage["max_episode_seconds"]),
                    "transitions_in_stage": int(self.transitions_in_stage),
                    "best_success_rate": val["success_rate"],
                    "note": "按设计**不**降低门槛晋级；此处仅记录以便分析该阶段的失败状态",
                    "elapsed_min": self._elapsed_min(),
                })
                self._write_history()
                print(f"[课程] 阶段 {stage['name']} 停滞（成功率 {val['success_rate']:.2f} < "
                      f"{self.promote_at}），按设计保持在该阶段，不降低标准晋级")
        return True

    # 让回调拿到归一化环境（构造时 vec_env 还没有；在 training_start 后可用）
    @property
    def _vn(self):
        return self._vec_normalize if self._vec_normalize is not None else self.model.get_env()

    def _on_training_end(self) -> None:
        self.history.append({
            "event": "end",
            "stop_reason": self.stop_reason,
            "stage": self._current_stage()["name"],
            "num_timesteps": int(self.num_timesteps),
            "elapsed_min": self._elapsed_min(),
        })
        self._write_history()


def _count_sources(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in episodes:
        k = str(e.get("termination_source"))
        out[k] = out.get(k, 0) + 1
    return out


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
                    tag_name: str = "best", cb: Optional["CurriculumCallback"] = None,
                    save_buffer: bool = True, sub_dir: Optional[Path] = None) -> Dict[str, str]:
    """按官方 checkpoint 的目录布局保存，使同一套加载代码可直接评估。

    SB3 的 ``model.save()`` **明确排除 replay buffer**（``_excluded_save_params``），
    因此若不单独保存，续训就会从**空经验池**重新开始 —— 这会让「断点续训」名不副实。
    这里额外保存：

    * ``<tag>_replay_buffer.pkl``：SB3 ``save_replay_buffer``（约 2.9 GiB，实测
      31,613 B/transition × 1e5）；
    * ``<tag>_curriculum_state.json``：课程阶段索引 / 阶段内计数 / 事件历史，
      供续训恢复（CurriculumCallback 新建时默认 stage_idx=0）。
    """
    import resource

    t0 = time.perf_counter()
    ckpt = (sub_dir if sub_dir is not None else run_dir / "checkpoint")
    ckpt.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Any] = {}

    model.save(str(ckpt / f"{tag_name}_model"))
    saved["model"] = str(ckpt / f"{tag_name}_model.zip")
    vec_normalize.save(str(ckpt / f"{tag_name}_env.zip"))
    saved["vec_normalize"] = str(ckpt / f"{tag_name}_env.zip")

    # 经验池：单独保存。体积大（~2.9 GiB）、耗时（~1-2 分钟），只在 5e4 间隔的
    # 检查点与最终保存时做；平时训练循环不保存。
    if save_buffer and getattr(model, "replay_buffer", None) is not None:
        bp = ckpt / f"{tag_name}_replay_buffer.pkl"
        t_buf = time.perf_counter()
        model.save_replay_buffer(str(bp))
        saved["replay_buffer"] = str(bp)
        saved["replay_buffer_size_bytes"] = int(bp.stat().st_size)
        saved["replay_buffer_save_seconds"] = float(time.perf_counter() - t_buf)

    # 课程状态：独立小文件，续训时读回
    if cb is not None:
        state = {
            "stage_idx": int(cb.stage_idx),
            "stage_name": cb._current_stage()["name"] if cb.stage_idx < len(cb.stages) else None,
            "stage_seconds": float(cb._current_stage()["max_episode_seconds"]) if cb.stage_idx < len(cb.stages) else None,
            "transitions_in_stage": int(cb.transitions_in_stage),
            "history": list(cb.history),
            "stop_reason": cb.stop_reason,
            "saved_at_num_timesteps": int(getattr(cb, "num_timesteps", 0)),
            "replay_buffer_pos": int(model.replay_buffer.pos)
            if hasattr(model.replay_buffer, "pos") else None,
            "replay_buffer_full": bool(model.replay_buffer.full)
            if hasattr(model.replay_buffer, "full") else None,
        }
        sp = ckpt / f"{tag_name}_curriculum_state.json"
        provenance.write_json(sp, state)
        saved["curriculum_state"] = str(sp)

    saved["peak_rss_gib"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1e6
    saved["total_save_seconds"] = float(time.perf_counter() - t0)

    if sub_dir is None:
        provenance.write_json(run_dir / "train_meta.json", meta)
        if not (run_dir / "locomotionFull.json").is_file():
            src = Path(cfg["checkpoint_dir"]) / "locomotionFull.json"
            if src.is_file():
                shutil.copy2(src, run_dir / "locomotionFull.json")
    return {k: v for k, v in saved.items() if isinstance(v, str)}


def load_curriculum_state(run_dir: Path, tag_name: str = "last") -> Optional[Dict[str, Any]]:
    """读回课程状态；文件不存在返回 None（旧产物没有这个文件）。"""
    p = run_dir / "checkpoint" / f"{tag_name}_curriculum_state.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 评估


def quick_eval(model, vec_normalize, spec: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    """在给定种子上跑一次确定性评估（单进程，不更新归一化统计）。

    **必须用与训练完全相同的 spec 构建环境**（同一终止判据、同一回合时长）。
    否则「验证集成功率」是在另一套规则下算出来的，课程升级就失去了意义。
    成功判定同时要求「走满本阶段时长」与配置里的速度/侧偏/倾角/数值判据。
    """
    from hemirl.research_env import ResearchEnvConfig as RC
    from hemirl.research_env import RewardSplitConfig as RS
    from hemirl.termination import TerminationConfig as TC
    from hemirl.rollout import root_state

    env_cfg = RC(
        termination=TC(
            kind="research",
            min_pelvis_height=spec["min_pelvis_height"],
            max_pelvis_up_tilt_deg=spec["max_pelvis_up_tilt_deg"],
            max_lateral_dev_from_ref_m=spec.get("max_lateral_dev_from_ref_m"),
            lateral_hold_s=float(spec.get("lateral_hold_s") or 0.0),
        ),
        max_episode_seconds=spec["max_episode_seconds"],
        reward_mode="split",
        reward_split=RS(w_survival=float(spec["w_survival"]),
                        include_official_healthy=bool(spec["include_official_healthy"])),
        extra_reward=HealthyRewardConfig(**spec["extra_reward"]) if spec.get("extra_reward") else None,
        keep_ledger=False,
        name="val",
    )
    success = dict(spec.get("success") or {})
    env, raw = build_research_env(env_cfg=env_cfg, checkpoint_dir=paths.checkpoint_dir("LocomotionFull"))
    vn = vec_normalize
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        steps = 0
        speeds: List[float] = []
        lats: List[float] = []
        tilts: List[float] = []
        while True:
            z = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            action, _ = model.predict(z, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            steps += 1
            rs = root_state(env.raw_env.model, env.raw_env.data, int(env.pelvis_id))
            speeds.append(float(rs["lin_vel"][0]))
            lats.append(abs(float(rs["pos"][1]) - env._ref_y()))
            tilts.append(float(env.up_tilt_deg()))
            if terminated or truncated:
                meta = info["termination"]
                break
        alive = float(meta["elapsed_time_s"])
        mean_vx = float(np.mean(speeds)) if speeds else 0.0
        checks = {
            "completed_planned_duration": meta["termination_source"] == "time_limit",
            "speed_ok": mean_vx >= float(success.get("min_mean_forward_speed", 0.0)),
            "lateral_ok": (max(lats) <= float(success.get("max_lateral_drift_m", 1e9))) if lats else False,
            "tilt_ok": (max(tilts) <= float(success.get("max_pelvis_tilt_deg", 180.0))) if tilts else False,
            "no_numeric_anomaly": not bool(meta["numeric_anomaly"]),
        }
        out.append({
            "seed": seed,
            "steps": steps,
            "alive_time_s": alive,
            "termination_source": meta["termination_source"],
            "completed": meta["termination_source"] == "time_limit",
            "mean_forward_speed": mean_vx,
            "max_lateral_dev_from_ref_m": float(max(lats)) if lats else None,
            "max_pelvis_tilt_deg": float(max(tilts)) if tilts else None,
            "success_checks": checks,
            "success": bool(all(checks.values())),
        })
    env.close()
    n = len(out)
    n_succ = sum(1 for r in out if r["success"])
    return {
        "n": n,
        "n_success": n_succ,
        "success_rate": float(n_succ / n) if n else 0.0,
        "complete_rate": float(sum(1 for r in out if r["completed"]) / n) if n else 0.0,
        "mean_alive_time_s": float(np.mean([r["alive_time_s"] for r in out])) if n else 0.0,
        "episodes": out,
        "seeds": seeds,
        "stage_seconds": spec["max_episode_seconds"],
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
    parser.add_argument("--resume-probe-z", type=str, default=None,
                        help="续训加载完成后、训练开始前，对给定 npz(z) 计算确定性动作并保存到 --resume-probe-out")
    parser.add_argument("--resume-probe-out", type=str, default=None,
                        help="--resume-probe-z 的输出路径（.npy）")
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
    # 课程状态提前读（stages 的起始阶段与探针都依赖它）
    resume_state = load_curriculum_state(run_dir, "last") if args.resume else None

    budget_transitions = args.max_transitions or cfg["budget"]["max_transitions"]
    budget_minutes = args.max_minutes or cfg["budget"]["max_minutes"]
    if args.check_only:
        budget_transitions = args.check_transitions
        budget_minutes = 10.0

    stages = cfg["curriculum"]["stages"]
    # 续训时 worker 仍按 stages[0] 构建，但探针直接用恢复后的阶段时长
    stage = stages[resume_state["stage_idx"]] if resume_state else stages[0]
    spec_stage = env_spec(cfg, stages[0])

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
        import resource

        last = run_dir / "checkpoint" / "last_model.zip"
        if not last.is_file():
            print(f"错误：--resume 需要 {last}")
            return 2
        t_resume = time.perf_counter()
        # 只加载 last，**不要**再走 load_official_model：
        # SB3 的 load 会用官方 checkpoint 里存下的 buffer_size 重建经验池
        # （1e6 → 29.4 GiB 瞬时分配），续训路径上这是无意义的开销与内存风险。
        model = type(model).load(str(last), env=vec_normalize, device="cpu", print_system_info=False)
        # SB3 save() 明确排除 replay buffer；load() 会重建一个空 buffer。
        # 这里从单独的 pkl 恢复经验池；文件不存在（旧产物）则显式标记热启动。
        buf_p = run_dir / "checkpoint" / "last_replay_buffer.pkl"
        if buf_p.is_file():
            t_buf = time.perf_counter()
            model.load_replay_buffer(str(buf_p))
            notes["resume_replay_buffer"] = {
                "file": str(buf_p),
                "size_bytes": int(buf_p.stat().st_size),
                "load_seconds": float(time.perf_counter() - t_buf),
                "buffer_pos_after": int(model.replay_buffer.pos),
                "buffer_full": bool(model.replay_buffer.full),
            }
            print(f"  经验池已从 {buf_p.name} 恢复（{buf_p.stat().st_size/1e9:.2f} GiB，"
                  f"pos={model.replay_buffer.pos} full={model.replay_buffer.full}）")
        else:
            notes["resume_replay_buffer"] = {
                "file": None,
                "note": "v1 权重热启动、新 buffer —— 未找到经验池文件，不声称恢复了经验池",
            }
            print("  未找到经验池文件：本次是 v1 权重热启动 + 新 buffer（不声称恢复经验池）")
        # 恢复后重新应用本项目的显式设置（load 会把 lr_schedule / amp 换成 checkpoint 里的值）
        model.dynsyn_weight_amp = float(cfg.get("dynsyn_weight_amp", 0.0))
        model.actor.dynsyn_layer.update_dynsyn_weight_amp(float(cfg.get("dynsyn_weight_amp", 0.0)))
        _lr = float(cfg["learning_rate"])
        model.lr_schedule = (lambda progress_remaining, _lr=_lr: _lr)
        for _opt in (model.actor.optimizer, model.critic.optimizer):
            for _g in _opt.param_groups:
                _g["lr"] = _lr
        resume_state = load_curriculum_state(run_dir, "last")
        notes["resumed_from"] = str(last)
        notes["resume_elapsed_seconds"] = float(time.perf_counter() - t_resume)
        notes["resume_peak_rss_gib"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1e6
        print(f"  已从 {last} 恢复（num_timesteps={model.num_timesteps}，"
              f"_n_updates={getattr(model, '_n_updates', '?')}，"
              f"课程状态={'有' if resume_state else '无'}）")

        # 续训保真探针：加载+重设 lr/amp **之后、训练之前**立即算一批确定性动作，
        # 供 scripts/check_resume.py 与断点前对比（验证续训路径没弄坏权重）。
        if args.resume_probe_z and args.resume_probe_out:
            import torch as _torch

            _z = np.load(args.resume_probe_z)["z"].astype(np.float32)
            with _torch.no_grad():
                _a = np.asarray(model.policy.actor(
                    _torch.as_tensor(_z), deterministic=True).numpy())
            np.save(args.resume_probe_out, _a)
            print(f"  续训保真探针已写: {args.resume_probe_out}")

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

    # --- 训练（CurriculumCallback 同时负责预算控制、课程阶段切换与周期检查点）
    cfg["_run_dir"] = str(run_dir)
    cb = CurriculumCallback(
        cfg=cfg,
        stages=stages,
        spec_builder=env_spec,
        budget_transitions=budget_transitions,
        budget_minutes=budget_minutes,
        vec_normalize=vec_normalize,
        resume_state=resume_state,
        run_dir=run_dir,
        checkpoint_every=int(cfg.get("checkpoint_every_transitions", 0)),
    )
    n_updates_before = int(getattr(model, "_n_updates", 0))
    t0 = time.perf_counter()
    print(f"\n开始训练…（上限 {budget_transitions} transitions / {budget_minutes} 分钟）；"
          f"课程阶段 = {[s['name'] for s in stages]}")
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
    final_stage = cb._current_stage()

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

    # --- 验证集评估（用**训练结束时所处阶段**的 spec，保证与训练规则一致）
    val_spec = env_spec(cfg, final_stage)
    val_res = quick_eval(model, vec_normalize, val_spec, list(cfg["curriculum"]["val_seeds"]))
    print(f"  验证集（{val_res['seeds']}，{val_spec['max_episode_seconds']:.0f}s 阶段）: "
          f"成功率={val_res['success_rate']:.2f} 走满率={val_res['complete_rate']:.2f} "
          f"平均存活={val_res['mean_alive_time_s']:.2f}s")

    n_this_run = int(done_transitions)
    n_total = int(model.num_timesteps)
    meta.update({
        "curriculum": {
            "stages": stages,
            "events": cb.history,
            "final_stage": final_stage["name"],
            "final_stage_seconds": float(final_stage["max_episode_seconds"]),
            "promoted_count": sum(1 for e in cb.history if e.get("event") == "promote"),
            "stalled": any(e.get("event") == "stalled" for e in cb.history),
        },
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
    saved = save_checkpoint(model, vec_normalize, run_dir, cfg, meta, "last", cb=cb)
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
        "final_stage": final_stage["name"],
        "final_stage_seconds": float(final_stage["max_episode_seconds"]),
        "n_promotions": sum(1 for e in cb.history if e.get("event") == "promote"),
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
