"""策略一致性验证：官方 SB3 路径 与 纯 torch 直读路径 必须逐元素一致。

## 为什么要做这个验证

评估结果的可信度取决于「我们读到的策略」是否就是「checkpoint 里训练出来的策略」。
本工作区有两条独立加载路径：

* **路径 A**：官方 ``stable_baselines3`` + 上游 ``DynSyn.SAC_DynSyn.Actor_DynSyn``；
* **路径 B**：``hemirl.policy.DirectActor``——用 ``policy.pth`` 的 state_dict 手工重建前向。

只要两者在**同一观测归一化结果**下、**确定性推理**时输出不一致，就说明至少有一条路径
实现错了。因此本模块把「一致」作为硬性验收项，不一致直接抛错（CLI 返回非零退出码）。

## 比对要做对的三件事

1. **同一归一化观测**：先过官方 ``VecNormalize.normalize_obs``，两条路径吃同一个数组；
   不能让其中一条自己再做归一化（否则比的是归一化实现，不是策略）。
2. **同一样本集合**：既有多个 ``reset`` 观测（覆盖不同初始相位），也有真实行走轨迹上的
   观测（覆盖策略实际访问到的状态分布）。
3. **失败要能定位**：逐层比对 ``latent_pi → mu → tanh(mu) → 组展开 → clamp``，
   找出**最早**出现差异的环节，而不是只看最终动作。

## 已知的早期缺陷（本模块的回归对象）

``latent_pi`` 由 SB3 的 ``create_mlp(features_dim, -1, net_arch, activation_fn)`` 构造。
``output_dim <= 0`` 时 SB3 会在**最后一层之后也加激活**，所以 ``latent_pi`` 以 ReLU 结尾。
早期实现漏掉了这个尾部 ReLU，导致两条路径最大动作差 ≈ 1.97（动作范围仅 [-1, 1]）。
:func:`layer_trace_direct` 提供 ``use_trailing_relu`` 开关，把这一缺陷**显式复现**出来，
作为「差异定位方法确实有效」的证据，而不是把阈值放宽掩盖它。

## dynsyn_weight_amp 的语义

``DynSynLayer.forward`` 在 ``dynsyn_weight_amp is None`` 时把 ``weight`` 置为全 1，
即**不做协同权重重标定**；checkpoint 的 ``data`` 里该字段为 ``null``。本模块用两条
可执行事实固定该语义：

* ``amp=None`` 与 ``amp=0.0`` 的输出必须**完全相同**（后者按公式恒等于 weight=1）；
* ``amp=0.05`` 的输出必须**不同**（证明这条分支是活的，不是死代码）。

其他 DynSyn 设置（如 ``amp=0.05``）只作为**命名实验**记录，不参与官方基线判定。
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hemirl import paths, provenance

#: 判定阈值：仅允许 float32 的数值噪声
DEFAULT_TOL = 1e-5

#: 上游对比字段（顺序即数据流顺序）
STAGE_ORDER = (
    "latent_pi",
    "mu",
    "tanh_mu",
    "weight_mean",
    "dynsyn_out",
    "final_action",
)


# ------------------------------------------------------------------ 逐层追踪


def _observations_as_tensor(obs: np.ndarray, device: str):
    import torch

    x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def layer_trace_sb3(stack, normalized_obs: np.ndarray) -> Dict[str, np.ndarray]:
    """用 forward hook 抓取官方 SB3 actor 的中间量（不重新实现前向）。"""
    import torch

    actor = stack.model.policy.actor
    x = _observations_as_tensor(normalized_obs, stack.device if hasattr(stack, "device") else "cpu")
    captured: Dict[str, Any] = {}
    handles = []

    def make(name):
        def hook(_module, _inp, out):
            captured[name] = out.detach().cpu().numpy().copy()

        return hook

    handles.append(actor.latent_pi.register_forward_hook(make("latent_pi")))
    handles.append(actor.mu.register_forward_hook(make("mu")))
    # dynsyn_layer.mu 是 DynSyn 的协同权重头（amp=None 时输出被丢弃，但仍记录以备审计）
    if hasattr(actor.dynsyn_layer, "mu"):
        handles.append(actor.dynsyn_layer.mu.register_forward_hook(make("weight_mean")))
    handles.append(actor.dynsyn_layer.register_forward_hook(make("dynsyn_out")))
    try:
        with torch.no_grad():
            action = actor(x, deterministic=True)
    finally:
        for h in handles:
            h.remove()
    captured["final_action"] = action.detach().cpu().numpy().copy()
    captured["tanh_mu"] = np.tanh(captured["mu"])
    return {k: v[0] if v.ndim >= 1 else v for k, v in captured.items()}


def layer_trace_direct(actor, normalized_obs: np.ndarray, use_trailing_relu: bool = True) -> Dict[str, np.ndarray]:
    """抓取路径 B 的中间量。

    Args:
        use_trailing_relu: ``True`` 为当前正确实现（``latent_pi`` 以 ReLU 结尾）；
            ``False`` 复现早期缺陷（最后一层后无激活），仅用于差异定位演示。
    """
    import torch

    x = _observations_as_tensor(normalized_obs, actor.device)
    with torch.no_grad():
        h = actor.latent_pi[:-1](x) if not use_trailing_relu else actor.latent_pi(x)
        mean = actor.mu(h)
        group_action = torch.tanh(mean)
        expanded = group_action[..., actor.group_of_muscle]
        final = torch.clamp(expanded, -1.0, 1.0)
    out = {
        "latent_pi": h.detach().cpu().numpy()[0],
        "mu": mean.detach().cpu().numpy()[0],
        "tanh_mu": group_action.detach().cpu().numpy()[0],
        "final_action": final.detach().cpu().numpy()[0],
    }
    return out


def first_divergent_stage(sb3: Dict[str, np.ndarray], direct: Dict[str, np.ndarray], tol: float) -> Optional[str]:
    """按数据流顺序返回第一个超差的环节名；全部在阈值内返回 ``None``。"""
    for stage in STAGE_ORDER:
        if stage not in sb3 or stage not in direct:
            continue
        if sb3[stage].shape != direct[stage].shape:
            return stage
        if float(np.max(np.abs(np.asarray(sb3[stage]) - np.asarray(direct[stage])))) > tol:
            return stage
    return None


# ------------------------------------------------------------------ amp 语义


def dynsyn_amp_semantics(stack, normalized_obs: np.ndarray) -> Dict[str, Any]:
    """用可执行事实固定 ``dynsyn_weight_amp`` 在评估时的语义。

    * ``None`` vs ``0.0``：必须逐元素相同（后者按公式恒为 weight=1）。
    * ``None`` vs ``0.05``：必须不同（证明重标定分支是活的）。
    """
    import torch

    actor = stack.model.policy.actor
    layer = actor.dynsyn_layer
    x = _observations_as_tensor(normalized_obs, "cpu")

    def run(amp):
        layer.update_dynsyn_weight_amp(amp)
        with torch.no_grad():
            out = actor(x, deterministic=True)
        return out.detach().cpu().numpy()

    original = layer.dynsyn_weight_amp
    try:
        a_none = run(None)
        a_zero = run(0.0)
        a_small = run(0.05)
    finally:
        layer.update_dynsyn_weight_amp(original)

    d_zero = float(np.max(np.abs(a_none - a_zero)))
    d_small = float(np.max(np.abs(a_none - a_small)))
    return {
        "amp_at_load": original,
        "max_abs_diff_none_vs_zero": d_zero,
        "max_abs_diff_none_vs_0p05": d_small,
        "none_equals_weight_one": bool(d_zero == 0.0),
        "amp_branch_is_live": bool(d_small > 1e-6),
        "interpretation": (
            "amp=None 时 DynSynLayer 把协同权重置为 1（不做重标定）；"
            "amp=0 与 amp=None 输出完全相同，amp=0.05 则产生差异，"
            "因此评估基线等价于「肌群动作按组展开」，未引入额外权重。"
        ),
    }


# ------------------------------------------------------------------ 主验证


def group_value_repeat_stats(a: np.ndarray, group_of_muscle: np.ndarray, groups=()) -> Dict[str, Any]:
    """校验「按组展开」是否真的做到**组内取值相同**。

    历史缺陷（已修正）：早期字段写作 ``np.array_equal(a, a[group_of_muscle])``。
    该式要求 ``g[g[i]] == g[i]``（``g`` 为 ``group_of_muscle``），也就是要求
    「肌肉下标恰好等于组号」这个巧合成立，因此在**完全正确**的展开上也会返回 False。
    实测：真实 checkpoint 的 138 个组**组内全部同值**（不同值的组数 = 0），
    而旧公式报 697/700 个元素“不等”。

    正确判据：每组取一个代表下标 ``first_of_group[g[i]]``，则应有
    ``a == a[first_of_group[group_of_muscle]]``。
    """
    a = np.asarray(a)
    g = np.asarray(group_of_muscle)
    n_groups = int(g.max()) + 1 if g.size else 0
    first_of_group = np.zeros(n_groups, dtype=int)
    for gi in range(n_groups):
        idx = np.where(g == gi)[0]
        first_of_group[gi] = int(idx.min()) if idx.size else 0
    ok = bool(np.allclose(a, a[first_of_group[g]]))
    n_bad_groups = 0
    if groups:
        for grp in groups:
            vals = a[np.asarray(grp, dtype=int)]
            if not np.allclose(vals, vals[0]):
                n_bad_groups += 1
    return {
        "group_value_repeat_ok": ok,
        "n_groups_with_mixed_values": int(n_bad_groups),
        "n_elements_mismatched_legacy_formula": int(np.sum(~np.isclose(a, a[g]))),
        "formula": "a == a[first_of_group[group_of_muscle]]（组内同值）",
        "legacy_formula_note": (
            "旧写法 a == a[group_of_muscle] 要求 g[g[i]]==g[i]，在正确展开上也会为 False，"
            "已废弃"
        ),
    }


@dataclass
class PolicyVerifyConfig:
    """策略一致性验证的输入。"""

    checkpoint_dir: Path = field(default_factory=lambda: paths.checkpoint_dir("LocomotionFull"))
    tol: float = DEFAULT_TOL
    n_reset: int = 6
    n_traj: int = 120
    traj_seeds: Tuple[int, ...] = (0, 1, 2)
    device: str = "cpu"
    run_amp_experiment: bool = True
    amp_experiment_value: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "tol": self.tol,
            "n_reset": self.n_reset,
            "n_traj": self.n_traj,
            "traj_seeds": list(self.traj_seeds),
            "device": self.device,
            "run_amp_experiment": self.run_amp_experiment,
            "amp_experiment_value": self.amp_experiment_value,
        }
        return d


def collect_observations(
    env,
    stack,
    cfg: PolicyVerifyConfig,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """收集两类观测：多个 reset 观测 + 真实行走轨迹上的观测。

    Returns:
        (normalized_obs_list, meta)
    """
    obs_list: List[np.ndarray] = []
    meta: Dict[str, Any] = {"n_reset": 0, "n_trajectory": 0, "trajectory_seeds": list(cfg.traj_seeds)}

    # --- 1) 多个 reset 观测（覆盖不同初始相位）
    for i in range(cfg.n_reset):
        obs, _ = env.reset(seed=1000 + i)
        obs_list.append(np.asarray(stack.normalize_obs(obs), dtype=np.float32))
    meta["n_reset"] = cfg.n_reset

    # --- 2) 真实行走轨迹上的观测（覆盖策略实际访问的状态分布）
    per_episode = max(1, int(np.ceil(cfg.n_traj / max(1, len(cfg.traj_seeds)))))
    n_traj = 0
    episode_lengths: List[int] = []
    for seed in cfg.traj_seeds:
        obs, _ = env.reset(seed=seed)
        n_recorded = 0
        for _step in range(per_episode):
            normalized = np.asarray(stack.normalize_obs(obs), dtype=np.float32)
            obs_list.append(normalized)
            n_recorded += 1
            action = stack.predict(normalized)
            obs, _r, terminated, truncated, _info = env.step(action)
            if bool(terminated) or bool(truncated):
                break
        episode_lengths.append(n_recorded)
        n_traj += n_recorded
    meta["n_trajectory"] = n_traj
    meta["episode_lengths"] = episode_lengths
    return obs_list, meta


def verify_policy(
    cfg: Optional[PolicyVerifyConfig] = None,
    *,
    out_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """执行策略一致性验证并返回报告字典。

    报告中的 ``passed`` 为最终判定；调用方应据此设置退出码。
    """
    from hemirl import policy as pol
    from hemirl.envs import OfficialEnvConfig, build_env
    from hemirl.wrappers import verify_wrapper_equivalence

    cfg = cfg or PolicyVerifyConfig()
    ckpt_dir = Path(cfg.checkpoint_dir)

    report: Dict[str, Any] = {
        "entry": "hemirl.policy_verify.verify_policy",
        "config": cfg.to_dict(),
        "tol": float(cfg.tol),
        "code_version": provenance.code_version(),
        "checkpoint_sha256": {
            "best_model.zip": provenance.file_sha256(ckpt_dir / "checkpoint" / "best_model.zip"),
            "best_env.zip": provenance.file_sha256(ckpt_dir / "checkpoint" / "best_env.zip"),
        },
    }

    # --- wrapper 等价性（动作接口不得漂移）
    wrapper = verify_wrapper_equivalence()
    report["wrapper_equivalence"] = wrapper
    if verbose:
        print(f"[wrapper] 等价={wrapper['equivalent']} max|diff|={wrapper['max_abs_diff']:.3e}")

    # --- 组展开语义（与上游 repeat_replace_x 对照）
    data = pol.load_policy_kwargs(ckpt_dir)
    groups = data["policy_kwargs"]["dynsyn"]
    exp = pol.build_group_expansion(groups)
    report["group_expansion"] = {
        "n_groups": exp["muscle_group_nums"],
        "muscle_dims": exp["muscle_dims"],
        "target_entropy_in_ckpt": data.get("target_entropy"),
        "unique_muscle_indices": int(np.unique(exp["group_of_muscle"]).size),
        "n_muscles_per_group_min": int(min(len(g) for g in groups)),
        "n_muscles_per_group_max": int(max(len(g) for g in groups)),
    }
    if verbose:
        print(f"[groups] {report['group_expansion']}")

    # --- 环境（官方配置，观测/动作接口不得改变）
    env_cfg = OfficialEnvConfig.from_checkpoint(ckpt_dir)
    env, _raw = build_env(env_cfg)

    try:
        stack = pol.load_sb3_stack(ckpt_dir, env, device=cfg.device, deterministic=True)
        report["sb3"] = {
            "policy_class": type(stack.model.policy).__name__,
            "actor_class": type(stack.model.policy.actor).__name__,
            "latent_pi_layers": _describe_sequential(stack.model.policy.actor.latent_pi),
            "vec_normalize_load_mode": stack.load_notes.get("vec_normalize_load_mode"),
            "float_schedule_fallbacks": stack.load_notes.get("float_schedule_fallbacks"),
        }
        if verbose:
            print(f"[sb3] {report['sb3']}")

        actor = pol.direct_actor_from_checkpoint(ckpt_dir, device=cfg.device)
        report["direct"] = {"latent_pi_layers": _describe_sequential(actor.latent_pi)}

        # --- amp 语义（可执行事实）
        amp_sem = dynsyn_amp_semantics(stack, env.reset(seed=0)[0])
        report["dynsyn_weight_amp_semantics"] = amp_sem
        report["dynsyn_weight_amp_at_load"] = amp_sem["amp_at_load"]
        if verbose:
            print(
                f"[amp] at_load={amp_sem['amp_at_load']} "
                f"none==0:{amp_sem['none_equals_weight_one']} live:{amp_sem['amp_branch_is_live']}"
            )

        # --- 样本收集
        obs_list, obs_meta = collect_observations(env, stack, cfg)
        report["samples"] = obs_meta
        report["n_samples"] = len(obs_list)
        if verbose:
            print(f"[samples] {report['n_samples']} 个（reset {obs_meta['n_reset']} + 轨迹 {obs_meta['n_trajectory']}）")

        # --- 逐样本比对
        per_sample: List[float] = []
        per_sample_stage: List[Dict[str, float]] = []
        for obs in obs_list:
            a_sb3 = stack.predict(obs)
            a_direct = actor(obs, deterministic=True)
            per_sample.append(float(np.max(np.abs(np.asarray(a_sb3) - np.asarray(a_direct)))))
            if len(per_sample_stage) < 5:
                tr_s = layer_trace_sb3(stack, obs)
                tr_d = layer_trace_direct(actor, obs)
                per_sample_stage.append(
                    {
                        s: float(np.max(np.abs(np.asarray(tr_s[s]) - np.asarray(tr_d[s]))))
                        for s in STAGE_ORDER
                        if s in tr_s and s in tr_d
                    }
                )

        max_diff = float(max(per_sample)) if per_sample else float("inf")
        report["max_abs_action_diff"] = max_diff
        report["mean_abs_action_diff"] = float(np.mean(per_sample)) if per_sample else float("inf")
        report["n_samples_exceeding_tol"] = int(sum(1 for d in per_sample if d > cfg.tol))
        report["per_sample_diffs_head"] = per_sample[:20]

        # --- 差异定位（分层）
        tr_sb3 = layer_trace_sb3(stack, obs_list[0])
        tr_direct = layer_trace_direct(actor, obs_list[0])
        stage_diffs = {
            s: float(np.max(np.abs(np.asarray(tr_sb3[s]) - np.asarray(tr_direct[s]))))
            for s in STAGE_ORDER
            if s in tr_sb3 and s in tr_direct
        }
        stage_shapes = {s: list(np.asarray(tr_sb3[s]).shape) for s in stage_diffs}
        report["layer_trace"] = {
            "stage_order": list(STAGE_ORDER),
            "stage_diffs": stage_diffs,
            "stage_shapes": stage_shapes,
            "first_divergent_stage": first_divergent_stage(tr_sb3, tr_direct, cfg.tol),
        }

        # --- 缺陷复现（证明定位方法有效，不作为通过条件）
        tr_bad = layer_trace_direct(actor, obs_list[0], use_trailing_relu=False)
        report["defect_reproduction"] = {
            "description": "复现早期缺陷：latent_pi 最后一层之后缺少 ReLU",
            "max_abs_action_diff_without_trailing_relu": float(
                np.max(np.abs(np.asarray(tr_sb3["final_action"]) - np.asarray(tr_bad["final_action"])))
            ),
            "first_divergent_stage": first_divergent_stage(tr_sb3, tr_bad, cfg.tol),
        }
        if verbose:
            print(f"[layer] stage_diffs={ {k: round(v, 3) for k, v in stage_diffs.items()} }")
            print(f"[defect] 缺尾部 ReLU 时 max|diff| = {report['defect_reproduction']['max_abs_action_diff_without_trailing_relu']:.4f}")

        # --- 动作统计（确认展开后仍落在 [-1,1] 且组内取值相同）
        a0 = np.asarray(stack.predict(obs_list[0]), dtype=float)
        env_action0 = env.action(a0)
        repeat_stats = group_value_repeat_stats(a0, exp["group_of_muscle"], groups)
        report["action_stats"] = {
            "dim": int(a0.shape[-1]),
            "min": float(a0.min()),
            "max": float(a0.max()),
            "mean": float(a0.mean()),
            "env_ctrl_min": float(env_action0.min()),
            "env_ctrl_max": float(env_action0.max()),
            "n_unique_group_values": int(np.unique(np.round(a0, 6)).size),
            "in_range": bool(a0.min() >= -1.0 and a0.max() <= 1.0),
            **repeat_stats,
        }

        # --- 命名实验：非官方 DynSyn 设置（不参与基线判定）
        named: Dict[str, Any] = {}
        if cfg.run_amp_experiment:
            layer = stack.model.policy.actor.dynsyn_layer
            original = layer.dynsyn_weight_amp
            diffs = []
            try:
                layer.update_dynsyn_weight_amp(cfg.amp_experiment_value)
                for obs in obs_list[: min(10, len(obs_list))]:
                    a_exp = np.asarray(stack.predict(obs), dtype=float)
                    a_base = np.asarray(actor(obs, deterministic=True), dtype=float)
                    diffs.append(float(np.max(np.abs(a_exp - a_base))))
            finally:
                layer.update_dynsyn_weight_amp(original)
            named["dynsyn_amp_nonzero"] = {
                "amp": cfg.amp_experiment_value,
                "n_samples": len(diffs),
                "max_abs_action_diff_vs_baseline": float(max(diffs)) if diffs else None,
                "note": "非官方设置，仅作命名实验记录，不参与官方基线一致性判定",
            }
        report["named_experiments"] = named

    finally:
        try:
            env.close()
        except Exception:
            pass

    # --- 判定
    checks = {
        "wrapper_equivalent": bool(wrapper["equivalent"]),
        "group_expansion_covers_all_muscles": report["group_expansion"]["unique_muscle_indices"]
        == report["group_expansion"]["n_groups"],
        "amp_none_equals_weight_one": bool(report["dynsyn_weight_amp_semantics"]["none_equals_weight_one"]),
        "amp_branch_is_live": bool(report["dynsyn_weight_amp_semantics"]["amp_branch_is_live"]),
        "action_within_range": bool(report["action_stats"]["in_range"]),
        "group_value_repeat_ok": bool(report["action_stats"]["group_value_repeat_ok"]),
        "no_layer_divergence": report["layer_trace"]["first_divergent_stage"] is None,
        "max_abs_action_diff_within_tol": max_diff <= cfg.tol,
    }
    report["checks"] = checks
    report["passed"] = bool(all(checks.values()))

    if out_path is not None:
        provenance.write_json(Path(out_path), report)

    if verbose:
        print("\n=== 判定 ===")
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(
            f"  max|a_sb3 - a_direct| = {max_diff:.3e}（阈值 {cfg.tol:.1e}，"
            f"样本 {report['n_samples']} 个）"
        )
        print(f"  结论: {'通过' if report['passed'] else '失败'}")
    return report


def _describe_sequential(seq) -> List[Dict[str, Any]]:
    """把 nn.Sequential 描述成可落盘的层清单（含激活函数）。"""
    out: List[Dict[str, Any]] = []
    for i, mod in enumerate(seq):
        name = type(mod).__name__
        if hasattr(mod, "weight") and hasattr(mod.weight, "shape"):
            out.append({"i": i, "type": name, "shape": list(mod.weight.shape)})
        else:
            out.append({"i": i, "type": name})
    return out


__all__ = [
    "PolicyVerifyConfig",
    "verify_policy",
    "layer_trace_sb3",
    "layer_trace_direct",
    "first_divergent_stage",
    "dynsyn_amp_semantics",
    "group_value_repeat_stats",
    "collect_observations",
    "DEFAULT_TOL",
    "STAGE_ORDER",
]
