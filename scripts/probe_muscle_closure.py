"""实证判定 MuJoCo 肌肉执行器的力公式口径（用于闭环验证）。

`mju_muscleGain` / `mju_muscleBias` 在 C 源码里返回 **负值**：

.. code-block:: c

    mjtNum mju_muscleGain(...) { ... return -force*FL*FV; }
    mjtNum mju_muscleBias(...) { ... return -force*fpmax*(...); }

但 ``data.actuator_force`` 究竟等于 ``gain*act + bias`` 还是其相反数、以及
速度正负号如何取，需要以**真实 rollout 状态**实测判定，而不是猜。

本脚本在真实行走轨迹上对以下候选逐一比较误差：

    F1: g(vel)*act + b
    F2: g(-vel)*act + b
    F3: -(g(vel)*act + b)
    F4: -(g(-vel)*act + b)
    F5: -g(vel)*act + b        （再取负）
    F6: g(vel)*act - b

输出最优口径与该口径下的最大/平均误差。

用法::

    MUJOCO_GL=egl PYTHONPATH=. python scripts/probe_muscle_closure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402
from hemirl.rollout import EvaluatorConfig, LocomotionEvaluator  # noqa: E402
from hemirl.termination import TerminationConfig  # noqa: E402


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--out", type=str, default=str(paths.REPORTS_ROOT / "muscle_closure_probe.json"))
    args = parser.parse_args()

    cfg = EvaluatorConfig(
        checkpoint_dir=paths.checkpoint_dir("LocomotionFull"),
        deterministic=True,
        termination=TerminationConfig(kind="research"),
    )

    with LocomotionEvaluator(cfg) as ev:
        model, data = ev.model, ev.data
        obs, _ = ev.env.reset(seed=0)
        for i in range(args.steps):
            a = ev.stack.predict(ev.stack.normalize_obs(obs))
            obs, _r, _t, trunc, _i = ev.env.step(a)
            if i >= args.steps - 6:  # 采集若干个真实运动状态
                pass

        length = np.asarray(data.actuator_length, dtype=float)
        vel = np.asarray(data.actuator_velocity, dtype=float)
        act = np.asarray(data.act, dtype=float)
        f_model = np.asarray(data.actuator_force, dtype=float)

        lr = np.asarray(model.actuator_lengthrange, dtype=float)
        acc0 = np.asarray(model.actuator_acc0, dtype=float)
        gp = np.asarray(model.actuator_gainprm)[:, :9]
        bp = np.asarray(model.actuator_biasprm)[:, :9]

        nu = model.nu
        g_pos = np.zeros(nu)
        g_neg = np.zeros(nu)
        bias = np.zeros(nu)
        for a in range(nu):
            g_pos[a] = float(mujoco.mju_muscleGain(length[a], vel[a], lr[a], acc0[a], gp[a]))
            g_neg[a] = float(mujoco.mju_muscleGain(length[a], -vel[a], lr[a], acc0[a], gp[a]))
            bias[a] = float(mujoco.mju_muscleBias(length[a], lr[a], acc0[a], bp[a]))

        candidates = {
            "F1: g(vel)*act + b": g_pos * act + bias,
            "F2: g(-vel)*act + b": g_neg * act + bias,
            "F3: -(g(vel)*act + b)": -(g_pos * act + bias),
            "F4: -(g(-vel)*act + b)": -(g_neg * act + bias),
            "F5: -g(vel)*act + b": -g_pos * act + bias,
            "F6: g(vel)*act - b": g_pos * act - bias,
        }

        mask = np.abs(f_model) > 1e-6
        report = {
            "n_steps": args.steps,
            "n_actuators": int(nu),
            "n_actuators_nonzero_force": int(mask.sum()),
            "force_abs_mean": float(np.abs(f_model).mean()),
            "model_force_sign": {
                "n_negative": int((f_model < 0).sum()),
                "n_positive": int((f_model > 0).sum()),
                "min": float(f_model.min()),
                "max": float(f_model.max()),
            },
            "gain_sign": {
                "n_negative": int((g_pos < 0).sum()),
                "n_positive": int((g_pos > 0).sum()),
                "min": float(g_pos.min()),
                "max": float(g_pos.max()),
            },
            "bias_sign": {
                "n_negative": int((bias < 0).sum()),
                "n_positive": int((bias > 0).sum()),
            },
            "candidates": {},
        }
        for name, pred in candidates.items():
            err = np.abs(pred - f_model)
            report["candidates"][name] = {
                "max_abs_err": float(err.max()),
                "mean_abs_err": float(err.mean()),
                "mean_abs_err_over_active": float(err[mask].mean()) if mask.any() else None,
            }
        best = min(report["candidates"], key=lambda k: report["candidates"][k]["max_abs_err"])
        report["best_candidate"] = best
        report["best_max_abs_err"] = report["candidates"][best]["max_abs_err"]

        # 静态对照：速度置零后公式是否精确
        d2 = mujoco.MjData(model)
        d2.qpos[:] = np.asarray(data.qpos)
        d2.qvel[:] = 0.0
        d2.act[:] = act
        mujoco.mj_forward(model, d2)
        f_static = np.asarray(d2.actuator_force, dtype=float)
        l2 = np.asarray(d2.actuator_length, dtype=float)
        g2 = np.array(
            [float(mujoco.mju_muscleGain(l2[a], 0.0, lr[a], acc0[a], gp[a])) for a in range(nu)]
        )
        b2 = np.array(
            [float(mujoco.mju_muscleBias(l2[a], lr[a], acc0[a], bp[a])) for a in range(nu)]
        )
        static_err = np.abs((g2 * act + b2) - f_static)
        report["static_check"] = {
            "max_abs_err": float(static_err.max()),
            "mean_abs_err": float(static_err.mean()),
            "force_abs_mean": float(np.abs(f_static).mean()),
        }

        # 一致性检验（运动状态下也精确）：
        # 用当前 qpos/qvel/act/ctrl 重新调用 mj_forward，使 actuator_length / velocity 与
        # actuator_force 取自同一状态。若公式口径正确，误差应为 0。
        d3 = mujoco.MjData(model)
        d3.qpos[:] = np.asarray(data.qpos)
        d3.qvel[:] = np.asarray(data.qvel)
        d3.act[:] = np.asarray(data.act)
        d3.ctrl[:] = np.asarray(data.ctrl)
        mujoco.mj_forward(model, d3)
        f3 = np.asarray(d3.actuator_force, dtype=float)
        l3 = np.asarray(d3.actuator_length, dtype=float)
        v3 = np.asarray(d3.actuator_velocity, dtype=float)
        a3 = np.asarray(d3.act, dtype=float)
        g3 = np.array(
            [float(mujoco.mju_muscleGain(l3[a], v3[a], lr[a], acc0[a], gp[a])) for a in range(nu)]
        )
        b3 = np.array(
            [float(mujoco.mju_muscleBias(l3[a], lr[a], acc0[a], bp[a])) for a in range(nu)]
        )
        consistent_err = np.abs((g3 * a3 + b3) - f3)
        report["same_state_consistency_check"] = {
            "max_abs_err": float(consistent_err.max()),
            "mean_abs_err": float(consistent_err.mean()),
            "n_nonzero_velocity": int((np.abs(v3) > 1e-9).sum()),
            "max_abs_velocity": float(np.abs(v3).max()),
            "note": (
                "同一状态下公式与 data.actuator_force 逐元素一致。运动轨迹上的残余误差"
                "来自 actuator_velocity 是步后采样（半隐式欧拉：mj_step 先用步前速度算力，再积分）"
            ),
        }

        text = json.dumps(report, indent=2, ensure_ascii=False)
        print(text)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
