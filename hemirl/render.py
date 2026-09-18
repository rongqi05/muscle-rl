"""渲染：导出评估短视频；渲染不可用时给出明确原因。

无显示环境下 MuJoCo 需要离屏后端。本机实测：

* ``MUJOCO_GL=egl`` 可用；
* ``MUJOCO_GL=osmesa`` **不可用**（系统未安装 ``libOSMesa``，PyOpenGL 报
  ``'NoneType' object has no attribute 'glGetError'``）。

因此默认建议 ``MUJOCO_GL=egl``。渲染失败时本模块不会让实验中断，而是返回失败原因，
并在调用方保存轨迹 npz 作为替代证据。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VideoResult:
    ok: bool
    path: Optional[str] = None
    n_frames: int = 0
    fps: float = 0.0
    error: Optional[str] = None
    backend: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "path": self.path,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "error": self.error,
            "backend": self.backend,
            "note": self.note,
        }


def _write_video(frames: List[np.ndarray], out_path: Path, fps: float) -> Optional[str]:
    """尝试写 mp4 / gif；返回失败原因（None 表示成功）。"""
    import imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(str(out_path), frames, fps=fps)
        return None
    except Exception as exc:  # 缺 ffmpeg 等
        alt = out_path.with_suffix(".gif")
        try:
            imageio.mimsave(str(alt), frames, fps=fps, loop=0)
            return None
        except Exception as exc2:
            return f"mp4 失败: {exc!r}; gif 失败: {exc2!r}"


def record_video(
    evaluator,
    seed: int,
    out_path: Path,
    spec=None,
    max_frames: Optional[int] = None,
) -> VideoResult:
    """跑一个 episode 并把它渲染成视频。

    Args:
        evaluator: 需要以 ``render_mode="rgb_array"`` 构造的 `LocomotionEvaluator`。
        seed: 回合种子。
        out_path: 输出路径（.mp4）。
        spec: 肌力配置；None 表示基准。
        max_frames: 最多录制的帧数。

    Returns:
        VideoResult
    """
    backend = os.environ.get("MUJOCO_GL", "(未设置，使用 MuJoCo 默认)")
    try:
        env = evaluator.env
        if env.render_mode != "rgb_array":
            return VideoResult(
                ok=False,
                error=f"render_mode={env.render_mode!r}，需要 'rgb_array'",
                backend=backend,
            )
        evaluator.scaler.reset()
        if spec is not None:
            if spec.mode != evaluator.scaler.mode:
                evaluator.scaler.mode = spec.mode
            evaluator.scaler.apply(spec)
        obs, _ = env.reset(seed=seed)
        frames: List[np.ndarray] = []
        n_max = evaluator.max_steps
        for _ in range(n_max + 1):
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))
            if max_frames is not None and len(frames) >= max_frames:
                break
            normalized = evaluator.stack.normalize_obs(obs)
            action = evaluator.stack.predict(normalized)
            obs, _r, terminated, truncated, _i = env.step(action)
            if terminated or truncated:
                # 终止的那一步也要渲染，因此先渲染再退出
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))
                break
        if not frames:
            return VideoResult(ok=False, error="未取到任何渲染帧", backend=backend)
        fps = float(env.metadata.get("render_fps", 50))
        err = _write_video(frames, Path(out_path), fps)
        if err is not None:
            return VideoResult(ok=False, error=err, n_frames=len(frames), backend=backend)
        out = Path(out_path)
        if not out.is_file():
            out = out.with_suffix(".gif")
        return VideoResult(
            ok=True,
            path=str(out),
            n_frames=len(frames),
            fps=fps,
            backend=backend,
            note="MuJoCo 离屏渲染（cameras: record_camera）",
        )
    except Exception as exc:
        return VideoResult(ok=False, error=f"{type(exc).__name__}: {exc}", backend=backend)


__all__ = ["VideoResult", "record_video"]
