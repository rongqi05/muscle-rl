"""短程中断/续训检查（阶段 A 交付）：验证 --resume 真的恢复了该恢复的东西。

覆盖：

1. **replay buffer**：保存/恢复后 `pos` 是否接续（而不是从 0 重来）；
2. **课程状态**：`curriculum_state.json` 被读回，历史里出现 ``resume`` 事件；
3. **确定性动作**：同一批归一化观测上，断点前后的 actor 输出逐位一致；
4. **归一化状态**：断点前后的 ``obs_rms.mean/var`` 逐位一致；
5. **优化器状态**：SB3 的 ``_get_torch_save_params`` 包含
   ``actor.optimizer/critic.optimizer``，断点前后的 Adam 状态应保留；
6. **文件与内存**：replay buffer 文件大小、峰值 RSS 被记录进 meta。

设计：分两段训练（各 600 transitions，buffer 用 2000 以控制体积），中间落盘
「探针」文件（固定 z 批量、归一化统计、优化器 state_dict），第二段 ``--resume``
后逐项对比。**不声称**并行环境的随机轨迹逐位复现。

运行::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/check_resume.py \
        --config configs/train_healthy_v3.json --tag resume_check \
        --transitions 600
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402

FAILURES: list = []


def check(name: str, ok: bool, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def run_train(args, tag: str, resume: bool, transitions: int, z_path: Path = None,
              out_path: Path = None) -> None:
    cmd = [
        sys.executable, str(paths.WORKSPACE_ROOT / "scripts" / "train_healthy.py"),
        "--config", args.config, "--tag", tag,
        "--max-transitions", str(transitions), "--max-minutes", "15",
        "--replay-buffer-size", "2000",
    ]
    if resume:
        cmd.append("--resume")
    if z_path is not None and out_path is not None:
        cmd += ["--resume-probe-z", str(z_path), "--resume-probe-out", str(out_path)]
    r = subprocess.run(cmd, cwd=str(paths.WORKSPACE_ROOT),
                       env={**os.environ, "PYTHONPATH": str(paths.WORKSPACE_ROOT),
                            "MUJOCO_GL": "egl"},
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"训练段失败（returncode={r.returncode}）")


def probe_segment(run_dir: Path, out: Path) -> None:
    """加载 last_model.zip，把关键量落盘到 out（供两段对比）。"""
    import torch

    from hemirl import policy as pol
    from hemirl.research_env import ResearchEnvConfig, build_research_env
    from hemirl.termination import TerminationConfig

    os.environ.setdefault("MUJOCO_GL", "egl")
    pol.install_numpy2_pickle_compat()
    paths.ensure_msgym_on_path()
    paths.ensure_dynsyn_scripts_on_path()
    paths.ensure_dynsyn_package_on_path()
    from DynSyn import SAC_DynSyn  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env, _raw = build_research_env(
        env_cfg=ResearchEnvConfig(termination=TerminationConfig(kind="research"),
                                  max_episode_seconds=4.0, keep_ledger=False, name="resume_probe"),
        checkpoint_dir=paths.checkpoint_dir("LocomotionFull"),
    )
    venv = DummyVecEnv([lambda: env])
    vn = VecNormalize.load(str(run_dir / "checkpoint" / "last_env.zip"), venv)
    with pol.patched_float_schedule():
        model = SAC_DynSyn.load(str(run_dir / "checkpoint" / "last_model.zip"),
                                env=vn, device="cpu", print_system_info=False)

    rng = np.random.default_rng(0)
    z = rng.normal(size=(8, 3601)).astype(np.float32)
    np.savez(str(out.with_suffix(".npz")), z=z)
    with torch.no_grad():
        a = np.asarray(model.policy.actor(torch.as_tensor(z), deterministic=True).numpy())
    probe = {
        "z": z,
        "a": a,
        "obs_mean": np.asarray(vn.obs_rms.mean).copy(),
        "obs_var": np.asarray(vn.obs_rms.var).copy(),
        "buffer_pos": int(model.replay_buffer.pos),
        "buffer_full": bool(model.replay_buffer.full),
        "num_timesteps": int(model.num_timesteps),
        "n_updates": int(model._n_updates),
        "log_ent_coef": float(model.log_ent_coef.item()),
        "dynsyn_weight_amp": model.dynsyn_weight_amp,
        "actor_opt_state": model.actor.optimizer.state_dict(),
        "critic_opt_state": model.critic.optimizer.state_dict(),
    }
    torch.save(probe, str(out))
    env.close()


def opt_summary(state: dict) -> str:
    try:
        p = state["state"][0]
        return (f"state_keys={sorted(p.keys())} "
                f"exp_avg_mean={p['exp_avg'].float().mean().item():.6f}")
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(paths.CONFIGS_ROOT / "train_healthy_v3.json"))
    parser.add_argument("--tag", default="resume_check")
    parser.add_argument("--transitions", type=int, default=600)
    args = parser.parse_args()

    run_dir = paths.RUNS_ROOT / "train" / args.tag
    import shutil

    shutil.rmtree(run_dir, ignore_errors=True)

    print("=== 第一段训练 ===")
    run_train(args, args.tag, resume=False, transitions=args.transitions)
    probe1 = run_dir / "probe_segment1.pt"
    probe_segment(run_dir, probe1)
    p1 = __import__("torch").load(str(probe1), weights_only=False)

    state1 = json.loads((run_dir / "checkpoint" / "last_curriculum_state.json").read_text(encoding="utf-8"))
    buf1 = run_dir / "checkpoint" / "last_replay_buffer.pkl"

    print("=== 第二段训练（--resume + 续训保真探针）===")
    z_npz = run_dir / "probe_segment1.npz"
    resume_actions = run_dir / "resume_probe_actions.npy"
    run_train(args, args.tag, resume=True, transitions=args.transitions,
              z_path=z_npz, out_path=resume_actions)
    probe2 = run_dir / "probe_segment2.pt"
    probe_segment(run_dir, probe2)
    p2 = __import__("torch").load(str(probe2), weights_only=False)

    meta = json.loads((run_dir / "train_meta.json").read_text(encoding="utf-8"))
    notes = meta.get("notes", {})
    state2 = json.loads((run_dir / "checkpoint" / "last_curriculum_state.json").read_text(encoding="utf-8"))
    hist = json.loads((run_dir / "curriculum_history.json").read_text(encoding="utf-8"))
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    n_envs = int(cfg.get("parallel_envs", 12))
    # buffer.pos 以「行」计（一行 = n_envs 个 transition）；预算到点的那一步不落盘
    expected_rows = max(1, args.transitions // n_envs - 1)

    print("=== 断点续训检查 ===")
    check("replay buffer 文件存在且非空", buf1.is_file() and buf1.stat().st_size > 0,
          f"{buf1.stat().st_size / 1e6:.1f} MiB")
    check("恢复时确实加载了 buffer（meta 有记录）",
          "resume_replay_buffer" in notes and notes["resume_replay_buffer"].get("file"),
          str(notes.get("resume_replay_buffer")))
    check("buffer 长度接续（第二段保存的 pos 前进约 transitions/n_envs 行）",
          int(state2.get("replay_buffer_pos", -1)) - int(state1.get("replay_buffer_pos", -2))
          >= expected_rows,
          f"{state1.get('replay_buffer_pos')} -> {state2.get('replay_buffer_pos')}（预期 ≥ +{expected_rows} 行）")
    check("恢复后 buffer.pos 与保存值一致",
          int(notes.get("resume_replay_buffer", {}).get("buffer_pos_after", -1)) == int(state1.get("replay_buffer_pos", -2)),
          str(notes.get("resume_replay_buffer", {})))
    check("课程状态文件被读回（历史含 resume 事件）",
          any(e.get("event") == "resume" for e in hist["events"]))
    a_resume = np.load(str(resume_actions))
    check("续训保真：加载后、训练前的确定性动作与断点前逐位一致",
          bool(np.allclose(p1["a"], a_resume, atol=1e-7)),
          f"max|Δa|={np.abs(p1['a'] - a_resume).max():.3e}")
    check("归一化统计逐位一致", bool(np.array_equal(p1["obs_mean"], p2["obs_mean"])
                                      and np.array_equal(p1["obs_var"], p2["obs_var"])))
    check("优化器 Adam 状态保留（actor）",
          sorted(p1["actor_opt_state"]["state"].keys()) == sorted(p2["actor_opt_state"]["state"].keys()),
          f"段1 {opt_summary(p1['actor_opt_state'])} | 段2 {opt_summary(p2['actor_opt_state'])}")
    check("num_timesteps 接续", int(p2["num_timesteps"]) == int(p1["num_timesteps"]) + args.transitions,
          f"{p1['num_timesteps']} -> {p2['num_timesteps']}")
    check("_n_updates 接续", int(p2["n_updates"]) >= int(p1["n_updates"]),
          f"{p1['n_updates']} -> {p2['n_updates']}")
    check("dynsyn_weight_amp 仍为固定值", float(p2["dynsyn_weight_amp"] or 0.0) == 0.0,
          str(p2["dynsyn_weight_amp"]))
    check("记录峰值内存与恢复耗时", float(notes.get("resume_peak_rss_gib", 0)) > 0
          and float(notes.get("resume_elapsed_seconds", 0)) > 0,
          f"{notes.get('resume_peak_rss_gib', 0):.2f} GiB, {notes.get('resume_elapsed_seconds', 0):.1f}s")

    out = {
        "tag": args.tag,
        "transitions_per_segment": args.transitions,
        "segment1": {"buffer_pos": state1.get("replay_buffer_pos"),
                     "num_timesteps": int(p1["num_timesteps"]), "n_updates": int(p1["n_updates"])},
        "segment2": {"buffer_pos": state2.get("replay_buffer_pos"),
                     "num_timesteps": int(p2["num_timesteps"]), "n_updates": int(p2["n_updates"])},
        "buffer_file_mib": round(buf1.stat().st_size / 1e6, 2),
        "resume_meta": notes.get("resume_replay_buffer"),
        "peak_rss_gib": notes.get("resume_peak_rss_gib"),
        "resume_seconds": notes.get("resume_elapsed_seconds"),
        "n_failures": len(FAILURES),
        "failures": list(FAILURES),
    }
    out["provenance"] = provenance.run_provenance(
        entry="scripts/check_resume.py", args=vars(args), extra={},
    )
    provenance.write_json(paths.REPORTS_ROOT / "resume_check.json", out)
    print(f"\n结果: {len(FAILURES)} 项失败 → {paths.REPORTS_ROOT / 'resume_check.json'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
