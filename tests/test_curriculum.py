"""课程阶段切换的回归测试（不需要 checkpoint / 不需要真实 MuJoCo 动力学）。

对应任务书第二阶段第 1 条：**「配置里列出 6/10/20 s 不算实现了课程」**。
本文件测的是「升级确实由独立验证集结果驱动、切换确实广播到 worker、不达标就不晋级」。

设计原则：只测行为，不复述实现。用桩对象替掉模型与向量环境，把
``quick_eval`` 替换成可控成功率的假函数，从而**确定性地**测出升级/停滞决策。

运行::

    PYTHONPATH=. python tests/test_curriculum.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402
from scripts import train_healthy as TH  # noqa: E402

RESULTS: list = []


def case(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
                RESULTS.append({"test": name, "ok": True, "error": None})
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                RESULTS.append({"test": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

        wrapper.__name__ = fn.__name__
        return wrapper

    return deco


# ------------------------------------------------------------------ 桩对象


class StubVecEnv:
    """记录 ``env_method`` 调用的向量环境桩。"""

    def __init__(self, n_envs: int = 4):
        self.num_envs = n_envs
        self.calls: List[tuple] = []

    def env_method(self, name, *args, **kwargs):
        self.calls.append((name, args))
        secs = float(args[0])
        return [{"max_episode_seconds": secs, "time_limit_s": secs,
                 "max_control_steps": int(round(secs / 0.02))}] * self.num_envs


class StubModel:
    """桩模型：``BaseCallback.training_env`` 实际取自 ``model.get_env()``。"""

    def __init__(self, num_timesteps: int = 0, venv=None):
        self.num_timesteps = num_timesteps
        self._venv = venv

    def get_env(self):
        return self._venv


class StubVN:
    """归一化环境桩：假评估不真的做归一化，只保留接口形状。"""

    def normalize_obs(self, obs, **kwargs):
        return obs

    def save(self, path):  # pragma: no cover - 本测试不做保存
        return path


#: 模块级假评估状态（``quick_eval`` 被整体替换，见文件末尾）。
_FAKE: Dict[str, Any] = {"rates": [0.0], "i": 0, "calls": []}


def _fake_quick_eval(model, vn, spec, seeds):
    """按 ``_FAKE["rates"]`` 依次返回可控成功率；记录 spec/seeds 供断言。"""
    i = min(_FAKE["i"], len(_FAKE["rates"]) - 1)
    _FAKE["i"] += 1
    rate = float(_FAKE["rates"][i])
    _FAKE["calls"].append({"spec": dict(spec), "seeds": list(seeds), "rate": rate})
    return {
        "n": len(seeds),
        "n_success": int(round(rate * len(seeds))),
        "success_rate": rate,
        "complete_rate": rate,
        "mean_alive_time_s": float(spec["max_episode_seconds"]) * rate,
        "episodes": [{"termination_source": "time_limit" if rate > 0.5 else "physical_fall"}],
        "seeds": list(seeds),
        "stage_seconds": spec["max_episode_seconds"],
    }


# 整体替换：本文件只测课程决策逻辑，不跑真实评估
TH.quick_eval = _fake_quick_eval


def make_cfg(tmp: Path, **cur_over) -> Dict[str, Any]:
    cur = {
        "stages": [
            {"name": "s4", "max_episode_seconds": 4.0, "min_transitions": 0},
            {"name": "s6", "max_episode_seconds": 6.0, "min_transitions": 100},
            {"name": "s10", "max_episode_seconds": 10.0, "min_transitions": 200},
        ],
        "promote_val_success_rate": 0.8,
        "val_seeds": [101, 102, 103],
        "val_every_transitions": 100,
        "stall_after_transitions": 300,
    }
    cur.update(cur_over)
    return {
        "curriculum": cur,
        "reward_split": {"w_survival": 100.0, "include_official_healthy": False},
        "extra_reward": {"enable": True},
        "eval": {"success": {"min_mean_forward_speed": 0.6, "max_lateral_drift_m": 0.35,
                             "max_pelvis_tilt_deg": 45.0}},
        "termination": {},
        "_run_dir": str(tmp),
    }


def make_cb(cfg, tmp: Path, rates: List[float], budget: int = 100000, minutes: float = 60.0):
    """构造回调；``quick_eval`` 由模块级假实现接管，按 ``rates`` 依次返回成功率。

    注意：替换必须在**整个测试期间**生效（而不是只在构造回调时），否则
    ``_on_step`` 触发验证时会走回真实评估（既慢又不可控）。
    """
    _FAKE["rates"] = [float(r) for r in rates]
    _FAKE["i"] = 0
    _FAKE["calls"] = []
    cb = TH.CurriculumCallback(
        cfg=cfg,
        stages=cfg["curriculum"]["stages"],
        spec_builder=TH.env_spec,
        budget_transitions=budget,
        budget_minutes=minutes,
        vec_normalize=StubVN(),
    )
    cb.model = StubModel(0, venv=StubVecEnv(n_envs=4))
    cb._on_training_start()
    return cb, _FAKE


def advance(cb, transitions: int, chunk: int = 4) -> List[bool]:
    """推进 ``transitions`` 个 transition（按 chunk 步进），返回每步是否继续。

    注意：SB3 的 ``BaseCallback.num_timesteps`` 是**普通属性**（在 ``on_step`` 里由
    ``model.num_timesteps`` 同步），不是 property——所以桩驱动必须两个都更新，
    否则回调看到的步数永远是 0。
    """
    outs = []
    left = transitions
    while left > 0:
        k = min(chunk, left)
        cb.model.num_timesteps += k
        cb.num_timesteps = cb.model.num_timesteps
        left -= k
        outs.append(bool(cb._on_step()))
    return outs


# ------------------------------------------------------------------ 测试


@case("C1 不达标不升级：验证成功率低于阈值时保持同一阶段，且不向 worker 广播")
def test_c1_no_promotion_below_threshold():
    with tempfile.TemporaryDirectory() as d:
        cfg = make_cfg(Path(d))
        cb, state = make_cb(cfg, Path(d), rates=[0.0])
        advance(cb, 400)
        assert cb.stage_idx == 0, cb.stage_idx
        assert cb.training_env.calls == [], cb.training_env.calls
        kinds = [e["event"] for e in cb.history]
        assert kinds.count("validate") >= 1, kinds
        assert "promote" not in kinds, kinds
        # 验证确实发生了，且用的是当前阶段（4 s）的 spec 与独立验证种子
        assert state["calls"], "quick_eval 未被调用"
        assert state["calls"][0]["spec"]["max_episode_seconds"] == 4.0, state["calls"][0]
        assert state["calls"][0]["seeds"] == [101, 102, 103], state["calls"][0]


@case("C2 达标才升级：成功率 ≥ 阈值时升级，并把新时长广播到全部 worker")
def test_c2_promotion_and_broadcast():
    with tempfile.TemporaryDirectory() as d:
        cfg = make_cfg(Path(d))
        # 把后续阶段的 min_transitions 抬高，使 40 个 transition 内**只**发生一次升级，
        # 从而隔离「升级本身」这件事（后续能否再升级由 C6 单独测）。
        cfg["curriculum"]["stages"][1]["min_transitions"] = 10**9
        cb, _state = make_cb(cfg, Path(d), rates=[0.9, 0.9, 0.9])
        advance(cb, 40)
        assert cb.stage_idx == 1, (cb.stage_idx, cb.history)
        assert len(cb.training_env.calls) == 1, cb.training_env.calls
        name, args = cb.training_env.calls[0]
        assert name == "configure_episode_seconds", cb.training_env.calls
        assert float(args[0]) == 6.0, cb.training_env.calls
        ev = [e for e in cb.history if e["event"] == "promote"]
        assert len(ev) == 1, ev
        assert ev[0]["from_stage"] == "s4" and ev[0]["to_stage"] == "s6", ev[0]
        assert ev[0]["n_workers"] == 4, ev[0]
        assert ev[0]["applied_in_workers"]["time_limit_s"] == 6.0, ev[0]
        assert ev[0]["applied_in_workers"]["max_control_steps"] == 300, ev[0]
        # 升级后阶段内计数必须清零，否则 min_transitions 立刻又满足
        assert cb.transitions_in_stage < 10**9, cb.transitions_in_stage
        # 验证发生在 s4 上，且用的是独立验证种子
        assert _state["calls"][0]["spec"]["max_episode_seconds"] == 4.0, _state["calls"][0]
        assert _state["calls"][0]["seeds"] == [101, 102, 103], _state["calls"][0]


@case("C3 min_transitions 是「最早可评估」门槛：未到步数不验证、不升级")
def test_c3_min_transitions_gate():
    with tempfile.TemporaryDirectory() as d:
        cfg = make_cfg(Path(d))
        cfg["curriculum"]["stages"][0]["min_transitions"] = 500
        cfg["curriculum"]["stages"][1]["min_transitions"] = 10**9  # 只允许升一级
        cb, state = make_cb(cfg, Path(d), rates=[0.99])
        advance(cb, 400)
        assert state["calls"] == [], state["calls"]
        assert cb.stage_idx == 0, cb.stage_idx
        advance(cb, 200)
        assert state["calls"], "超过 min_transitions 后应触发验证"
        assert cb.stage_idx == 1, cb.stage_idx
        assert state["calls"][0]["spec"]["max_episode_seconds"] == 4.0, state["calls"][0]


@case("C4 停滞不晋级：长时间不达标只记录 stalled，不得降低门槛通过")
def test_c4_stall_records_but_never_promotes():
    with tempfile.TemporaryDirectory() as d:
        cfg = make_cfg(Path(d))
        cb, _state = make_cb(cfg, Path(d), rates=[0.0])
        advance(cb, 1000)
        assert cb.stage_idx == 0, cb.stage_idx
        assert cb.training_env.calls == [], cb.training_env.calls
        stalls = [e for e in cb.history if e["event"] == "stalled"]
        assert len(stalls) == 1, cb.history
        assert stalls[0]["stage"] == "s4", stalls[0]
        assert "不" in stalls[0]["note"], stalls[0]


@case("C5 预算仍然生效：transition 与墙钟两个上限都能停止训练")
def test_c5_budget_still_applies():
    with tempfile.TemporaryDirectory() as d:
        cfg = make_cfg(Path(d))
        cb, _s = make_cb(cfg, Path(d), rates=[0.9], budget=100, minutes=60.0)
        outs = advance(cb, 200)
        assert False in outs, "到达 transition 预算应返回 False"
        assert cb.stop_reason == "到达 transition 预算", cb.stop_reason

        cb2, _s2 = make_cb(cfg, Path(d), rates=[0.9], budget=10**9, minutes=0.0)
        assert cb2._on_step() is False, "墙钟预算为 0 应立即停止"
        assert cb2.stop_reason == "到达墙钟预算", cb2.stop_reason


@case("C6 最高阶段不再尝试升级，但仍继续验证记录")
def test_c6_last_stage_no_promotion():
    with tempfile.TemporaryDirectory() as d:
        cfg = make_cfg(Path(d))
        cb, state = make_cb(cfg, Path(d), rates=[0.99])
        advance(cb, 120)   # s4 -> s6
        advance(cb, 220)   # s6 -> s10
        assert cb.stage_idx == 2, (cb.stage_idx, cb.history)
        n_calls = len(cb.training_env.calls)
        advance(cb, 400)
        assert cb.stage_idx == 2, cb.stage_idx
        assert len(cb.training_env.calls) == n_calls, "最高阶段不得再广播"
        assert not any(e["event"] == "promote" and e["to_stage"] == "s10"
                       for e in cb.history if e["event"] == "promote"
                       and e.get("from_stage") == "s10"), cb.history


@case("C7 课程历史落盘：curriculum_history.json 含 start/validate/promote/end")
def test_c7_history_written():
    with tempfile.TemporaryDirectory() as d:
        run_dir = Path(d)
        cfg = make_cfg(run_dir)
        cb, _s = make_cb(cfg, run_dir, rates=[0.9])
        advance(cb, 120)
        cb.stop_reason = "测试"
        cb._on_training_end()
        p = run_dir / "curriculum_history.json"
        assert p.is_file(), "课程历史必须落盘"
        h = json.loads(p.read_text(encoding="utf-8"))
        kinds = [e["event"] for e in h["events"]]
        for k in ("start", "validate", "promote", "end"):
            assert k in kinds, (k, kinds)
        assert h["val_seeds"] == cfg["curriculum"]["val_seeds"], h["val_seeds"]


@case("C8 env_spec 传递完整性：终止项、奖励权重、质量阈值都必须进入 spec")
def test_c8_env_spec_completeness():
    cfg = make_cfg(Path(tempfile.mkdtemp()))
    cfg["termination"] = {"max_lateral_dev_from_ref_m": None, "lateral_hold_s": 0.0}
    cfg["reward_split"] = {"w_survival": 20.0, "include_official_healthy": False}
    spec = TH.env_spec(cfg, {"name": "s6", "max_episode_seconds": 6.0})
    assert spec["max_episode_seconds"] == 6.0, spec
    assert spec["w_survival"] == 20.0, spec
    assert spec["max_lateral_dev_from_ref_m"] is None, spec
    # 质量阈值必须带进 spec，否则验证集的「达标」会是另一套标准
    assert spec["success"]["max_lateral_drift_m"] == 0.35, spec
    assert spec["success"]["min_mean_forward_speed"] == 0.6, spec
    import pickle

    pickle.loads(pickle.dumps(spec))  # 必须可 pickle（SubprocVecEnv 要求）


@case("C9 真实配置：课程最终阶段就是验收时长，且验证种子与测试种子不相交")
def test_c9_real_configs():
    for name in ("train_healthy_v1.json", "train_healthy_v2.json", "train_healthy_v3.json"):
        p = paths.CONFIGS_ROOT / name
        if not p.is_file():
            continue
        cfg = json.loads(p.read_text(encoding="utf-8"))
        stages = cfg["curriculum"]["stages"]
        secs = [s["max_episode_seconds"] for s in stages]
        assert secs == sorted(secs), (name, secs)
        assert secs[-1] == float(cfg["eval"]["episode_seconds"]), (name, secs, cfg["eval"]["episode_seconds"])
        val = set(cfg["curriculum"]["val_seeds"])
        test = set(cfg["eval"]["test_seeds"])
        assert not (val & test), (name, val & test)
        # 所有阶段名唯一
        names = [s["name"] for s in stages]
        assert len(names) == len(set(names)), names


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    n_ok = sum(1 for r in RESULTS if r["ok"])
    print(f"\n课程学习回归: {n_ok}/{len(RESULTS)} 通过")
    out = paths.REPORTS_ROOT / "unit_tests_curriculum.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(RESULTS), "n_pass": n_ok, "results": RESULTS},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
