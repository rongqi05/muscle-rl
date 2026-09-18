"""配对评估：原始官方策略（A）vs 正常肌力微调策略（B）。

## 协议（固定，不根据测试结果回调）

* 同一批**配对种子**、同一评估环境（``build_research_env``，研究终止语义）、
  同一确定性推理设置、同一归一化统计（冻结，不更新）；
* 验证种子（课程升级/模型选择用，如 101–105）与**最终测试种子**（201–220）分离；
* 每个策略跑 20 s 回合，全程不中途 reset、不覆盖根节点、不加稳定力。

## 成功判据（必须同时满足「在走」与「没倒」）

只按存活时间判定是不够的——原地站立也能「存活」。因此每个回合同时要求：

1. 走完全部规定时长（``truncated`` 于 ``time_limit``，不是跌倒）；
2. 平均前进速度 ≥ 阈值；
3. 侧向偏移 ≤ 阈值；
4. 骨盆最大倾角 ≤ 阈值；
5. 无 ``numeric_anomaly``。

并额外报告**退化行为检测**：完成回合中平均前进速度低于 ``degenerate_speed`` 视为「站着不走」。

工程门槛（本项目的阶段性标准，不代表临床有效性）：
独立测试 20 回合中 **≥18 个** 满足上述全部条件。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_healthy.py \\
        --policies A=artifacts/checkpoints/LocomotionFull B=runs/train/healthy_v1 \\
        --seeds 201:220 --tag healthy_vs_official
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths, provenance  # noqa: E402
from hemirl.research_env import ResearchEnvConfig, build_research_env  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402

FOOT_BODIES = ("calcn_r", "calcn_l", "toes_r", "toes_l")


class PairedEvaluator:
    """一个策略在固定协议下的评估器（自带足部滑动/动作饱和/退化行为统计）。"""

    def __init__(self, name: str, checkpoint_dir: Path, cfg: Dict[str, Any]):
        import mujoco

        from hemirl import policy as pol

        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cfg = cfg
        self.mujoco = mujoco

        env_cfg = ResearchEnvConfig(
            termination=TerminationConfig(kind="research", min_pelvis_height=0.55,
                                          max_pelvis_up_tilt_deg=60.0),
            max_episode_seconds=float(cfg["episode_seconds"]),
            keep_ledger=False,
            name=f"eval_{name}",
        )
        self.env, self.raw = build_research_env(env_cfg=env_cfg, checkpoint_dir=self.checkpoint_dir)
        self.model = self.raw.model
        self.data = self.raw.data
        self.stack = pol.load_sb3_stack(self.checkpoint_dir, self.env, deterministic=True)
        self.pelvis_id = int(self.env.pelvis_id)
        self.foot_ids = {
            n: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)) for n in FOOT_BODIES
        }
        # 官方基准的前进速度（目标速度误差用）
        self.ref_speed = float(self.raw.trajectory.get_trajectory_properties(0)[1])

    # ------------------------------------------------------------ 指标

    def _foot_slip_and_support(self) -> Tuple[float, int]:
        """返回 (足部滑动代理量, 触地足数)，只计法向力 > 20 N 的接触。"""
        buf = np.zeros(6, dtype=np.float64)
        geom_f: Dict[int, float] = {}
        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]
            self.mujoco.mj_contactForce(self.model, self.data, i, buf)
            f = float(abs(buf[0]))
            for gid in (int(con.geom1), int(con.geom2)):
                if f > geom_f.get(gid, 0.0):
                    geom_f[gid] = f
        slip = 0.0
        n_loaded = 0
        for bid in self.foot_ids.values():
            if bid < 0:
                continue
            adr, num = int(self.model.body_geomadr[bid]), int(self.model.body_geomnum[bid])
            loaded = any(geom_f.get(adr + g, 0.0) > 20.0 for g in range(num))
            if loaded:
                n_loaded += 1
                v = np.asarray(self.data.cvel[bid][3:6], dtype=float)
                slip = max(slip, float(np.linalg.norm(v)))
        return slip, n_loaded

    def run_episode(self, seed: int) -> Dict[str, Any]:
        obs, _ = self.env.reset(seed=seed)
        root_start = np.asarray(self.data.xpos[self.pelvis_id], dtype=float).copy()

        slip_vals: List[float] = []
        speed_vals: List[float] = []
        sat_vals: List[float] = []
        act_vals: List[float] = []
        lateral_dev: List[float] = []
        tilt_vals: List[float] = []
        standing_steps = 0
        reward_sum = 0.0

        from hemirl.rollout import root_state

        while True:
            z = np.asarray(self.stack.normalize_obs(obs), dtype=np.float32)
            action = np.asarray(self.stack.predict(z), dtype=float)
            obs, reward, terminated, truncated, info = self.env.step(action)

            rs = root_state(self.model, self.data, self.pelvis_id)
            slip, n_loaded = self._foot_slip_and_support()
            vx = float(rs["lin_vel"][0])
            slip_vals.append(slip)
            speed_vals.append(vx)
            sat_vals.append(float(np.mean(np.abs(action) > 0.99)))
            act_vals.append(float(np.mean(np.asarray(self.data.act, dtype=float))))
            lateral_dev.append(abs(float(rs["pos"][1]) - self.env._ref_y()))
            tilt_vals.append(float(self.env.up_tilt_deg()))
            reward_sum += float(reward)
            # 退化检测：双足都在支撑且前进速度很低 → 站着不走
            if n_loaded >= 2 and abs(vx) < float(self.cfg["degenerate_speed"]):
                standing_steps += 1

            if terminated or truncated:
                meta = info["termination"]
                break

        root_end = np.asarray(self.data.xpos[self.pelvis_id], dtype=float).copy()
        disp = root_end - root_start
        alive = float(meta["elapsed_time_s"])
        n = max(1, int(meta["n_control_steps"]))
        mean_vx = float(np.mean(speed_vals)) if speed_vals else 0.0

        success_cfg = self.cfg["success"]
        completed = meta["termination_source"] == "time_limit"
        checks = {
            "completed_planned_duration": bool(completed),
            "speed_ok": bool(mean_vx >= float(success_cfg["min_mean_forward_speed"])),
            "lateral_ok": bool(max(lateral_dev) <= float(success_cfg["max_lateral_drift_m"]))
            if lateral_dev
            else False,
            "tilt_ok": bool(max(tilt_vals) <= float(success_cfg["max_pelvis_tilt_deg"])) if tilt_vals else False,
            "no_numeric_anomaly": not bool(meta["numeric_anomaly"]),
        }
        return {
            "policy": self.name,
            "seed": seed,
            "steps": n,
            "alive_time_s": alive,
            "termination_source": meta["termination_source"],
            "termination_reason": meta["termination_reason"],
            "completed": bool(completed),
            "numeric_anomaly": bool(meta["numeric_anomaly"]),
            "physical_fall": bool(meta["physical_fall"]),
            "forward_displacement_x": float(disp[0]),
            "lateral_drift_y": float(disp[1]),
            "mean_forward_speed": mean_vx,
            "target_speed_error": abs(mean_vx - self.ref_speed),
            "reference_forward_speed": self.ref_speed,
            "max_lateral_dev_from_ref_m": float(max(lateral_dev)) if lateral_dev else None,
            "max_pelvis_tilt_deg": float(max(tilt_vals)) if tilt_vals else None,
            "foot_slip_mean_mps": float(np.mean(slip_vals)) if slip_vals else None,
            "foot_slip_max_mps": float(np.max(slip_vals)) if slip_vals else None,
            "activation_mean": float(np.mean(act_vals)) if act_vals else None,
            "action_saturation_mean": float(np.mean(sat_vals)) if sat_vals else None,
            "standing_steps": int(standing_steps),
            "standing_fraction": float(standing_steps / n),
            "reward_sum": reward_sum,
            "success_checks": checks,
            "success": bool(all(checks.values())),
            "degenerate_stood_still": bool(completed and mean_vx < float(self.cfg["degenerate_speed"])),
        }

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def aggregate(rows: List[Dict[str, Any]], gate: Dict[str, Any]) -> Dict[str, Any]:
    n = len(rows)
    comp = [r for r in rows if r["completed"]]
    fell = [r for r in rows if r["physical_fall"]]
    num = [r for r in rows if r["numeric_anomaly"]]
    succ = [r for r in rows if r["success"]]
    deg = [r for r in rows if r["degenerate_stood_still"]]
    return {
        "n_episodes": n,
        "complete_rate": len(comp) / n if n else 0.0,
        "success_rate": len(succ) / n if n else 0.0,
        "n_completed": len(comp),
        "n_fell": len(fell),
        "n_numeric_anomaly": len(num),
        "n_time_truncated": len(comp),
        "n_degenerate_stood_still": len(deg),
        "mean_alive_time_s": float(np.mean([r["alive_time_s"] for r in rows])) if n else 0.0,
        "mean_forward_speed": float(np.mean([r["mean_forward_speed"] for r in rows])) if n else 0.0,
        "mean_target_speed_error": float(np.mean([r["target_speed_error"] for r in rows])) if n else 0.0,
        "mean_abs_lateral_drift_m": float(np.mean([abs(r["lateral_drift_y"]) for r in rows])) if n else 0.0,
        "mean_max_lateral_dev_m": float(
            np.mean([r["max_lateral_dev_from_ref_m"] for r in rows if r["max_lateral_dev_from_ref_m"] is not None])
        )
        if any(r["max_lateral_dev_from_ref_m"] is not None for r in rows)
        else 0.0,
        "mean_max_pelvis_tilt_deg": float(
            np.mean([r["max_pelvis_tilt_deg"] for r in rows if r["max_pelvis_tilt_deg"] is not None])
        )
        if any(r["max_pelvis_tilt_deg"] is not None for r in rows)
        else 0.0,
        "mean_foot_slip_mps": float(
            np.mean([r["foot_slip_mean_mps"] for r in rows if r["foot_slip_mean_mps"] is not None])
        )
        if any(r["foot_slip_mean_mps"] is not None for r in rows)
        else 0.0,
        "mean_activation": float(np.mean([r["activation_mean"] for r in rows])) if n else 0.0,
        "mean_action_saturation": float(np.mean([r["action_saturation_mean"] for r in rows])) if n else 0.0,
        "completion_by_source": _count(rows, "termination_source"),
        "gate": {
            **gate,
            "n_ok": len(succ),
            "passed": bool(len(succ) >= int(gate["n_required"])),
        },
    }


def _count(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[str(r[key])] = out.get(str(r[key]), 0) + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="配对评估：官方 vs 微调")
    parser.add_argument("--policies", nargs="+", required=True,
                        help="形如 A=<dir> B=<dir>；A 应为原始官方 checkpoint")
    parser.add_argument("--config", type=str, default=str(paths.CONFIGS_ROOT / "train_healthy_v1.json"))
    parser.add_argument("--seeds", type=str, default=None, help="如 201:220 或 201 202 203")
    parser.add_argument("--tag", type=str, default="healthy_vs_official")
    parser.add_argument("--out-root", type=str, default=str(paths.RUNS_ROOT))
    args = parser.parse_args()

    cfgfile = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ev_cfg = dict(cfgfile["eval"])
    if args.seeds:
        if ":" in args.seeds:
            lo, hi = args.seeds.split(":")
            seeds = list(range(int(lo), int(hi) + 1))
        else:
            seeds = [int(s) for s in args.seeds.split()]
    else:
        seeds = list(ev_cfg["test_seeds"])
    ev_cfg["episode_seconds"] = float(ev_cfg.get("episode_seconds", 20.0))

    policies = []
    for item in args.policies:
        name, _, d = item.partition("=")
        if not d:
            raise SystemExit(f"--policies 需要 name=dir 形式，收到 {item!r}")
        policies.append((name, Path(d)))

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "entry": "scripts/eval_healthy.py",
        "protocol": {
            "seeds": seeds,
            "episode_seconds": ev_cfg["episode_seconds"],
            "deterministic": True,
            "vec_normalize_updated": False,
            "success_criteria": ev_cfg["success"],
            "gate": ev_cfg["gate"],
            "no_mid_episode_reset": True,
            "no_root_override": True,
            "no_stabilizing_force": True,
            "paired_seeds": True,
        },
        "policies": {},
    }

    for name, d in policies:
        print(f"\n=== 策略 {name}: {d} ===")
        t0 = time.perf_counter()
        ev = PairedEvaluator(name, d, ev_cfg)
        rows: List[Dict[str, Any]] = []
        for seed in seeds:
            r = ev.run_episode(seed)
            rows.append(r)
            print(
                f"  seed={seed:3d} alive={r['alive_time_s']:5.2f}s src={r['termination_source']:<12s} "
                f"v_x={r['mean_forward_speed']:+.3f} lat_dev={r['max_lateral_dev_from_ref_m']:.3f} "
                f"tilt={r['max_pelvis_tilt_deg']:5.1f}° slip={r['foot_slip_mean_mps']:.2f} "
                f"sat={r['action_saturation_mean']:.3f} success={r['success']}"
            )
        ev.close()
        agg = aggregate(rows, ev_cfg["gate"])
        report["policies"][name] = {
            "checkpoint_dir": str(d),
            "wall_time_s": time.perf_counter() - t0,
            "aggregate": agg,
            "episodes": rows,
        }
        print(f"  完成率={agg['complete_rate']:.2f} 成功率={agg['success_rate']:.2f} "
              f"跌倒={agg['n_fell']} 数值异常={agg['n_numeric_anomaly']} "
              f"退化站立={agg['n_degenerate_stood_still']}")

    # 配对对比
    if len(report["policies"]) >= 2:
        names = list(report["policies"])
        a, b = names[0], names[1]
        ra = {r["seed"]: r for r in report["policies"][a]["episodes"]}
        rb = {r["seed"]: r for r in report["policies"][b]["episodes"]}
        common = sorted(set(ra) & set(rb))
        report["paired_comparison"] = {
            "A": a, "B": b, "n_paired": len(common),
            "delta_alive_time_s": float(np.mean([rb[s]["alive_time_s"] - ra[s]["alive_time_s"] for s in common])),
            "delta_forward_speed": float(np.mean([rb[s]["mean_forward_speed"] - ra[s]["mean_forward_speed"] for s in common])),
            "delta_max_lateral_dev_m": float(
                np.mean([rb[s]["max_lateral_dev_from_ref_m"] - ra[s]["max_lateral_dev_from_ref_m"] for s in common])
            ),
            "delta_max_tilt_deg": float(
                np.mean([rb[s]["max_pelvis_tilt_deg"] - ra[s]["max_pelvis_tilt_deg"] for s in common])
            ),
            "delta_action_saturation": float(
                np.mean([rb[s]["action_saturation_mean"] - ra[s]["action_saturation_mean"] for s in common])
            ),
            "delta_success_rate": float(
                report["policies"][b]["aggregate"]["success_rate"]
                - report["policies"][a]["aggregate"]["success_rate"]
            ),
            "per_seed": [
                {"seed": s, "alive_A": ra[s]["alive_time_s"], "alive_B": rb[s]["alive_time_s"],
                 "success_A": ra[s]["success"], "success_B": rb[s]["success"]}
                for s in common
            ],
        }

    provenance.write_json(out_dir / "eval_report.json", report)
    provenance.write_json(out_dir / "run.json", provenance.run_provenance(
        entry="scripts/eval_healthy.py", args=vars(args),
        extra={"protocol": report["protocol"]},
    ))

    print("\n=== 汇总 ===")
    for name in report["policies"]:
        agg = report["policies"][name]["aggregate"]
        print(f"  {name}: 完成率={agg['complete_rate']:.2f} 成功率={agg['success_rate']:.2f} "
              f"平均存活={agg['mean_alive_time_s']:.2f}s 平均前进速度={agg['mean_forward_speed']:.3f} "
              f"最大侧偏={agg['mean_max_lateral_dev_m']:.3f}m 最大倾角={agg['mean_max_pelvis_tilt_deg']:.1f}° "
              f"门槛={'达到' if agg['gate']['passed'] else '未达到'}"
              f"({agg['gate']['n_ok']}/{agg['gate']['n_required']})")
    if "paired_comparison" in report:
        pc = report["paired_comparison"]
        print(f"  配对差值({pc['B']} - {pc['A']}，n={pc['n_paired']}): "
              f"存活 {pc['delta_alive_time_s']:+.2f}s, 速度 {pc['delta_forward_speed']:+.3f} m/s, "
              f"侧偏 {pc['delta_max_lateral_dev_m']:+.3f} m, 成功率 {pc['delta_success_rate']:+.2f}")
    print(f"\n[saved] {out_dir / 'eval_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
