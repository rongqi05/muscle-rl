"""运行 provenance：仓库 commit、checkpoint 来源与哈希、依赖版本、随机种子。

每次评估/扫描都会写一个 ``run.json``，保证结果可追溯。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hemirl import paths


def _git(args: list, cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "--no-pager", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def repo_info(root: Path, label: str) -> Dict[str, Any]:
    """返回某个仓库/目录的 commit 与脏状态。"""
    root = Path(root)
    if not (root / ".git").exists():
        return {"label": label, "path": str(root), "is_git": False}
    info: Dict[str, Any] = {
        "label": label,
        "path": str(root),
        "is_git": True,
        "commit": _git(["rev-parse", "HEAD"], root),
        "commit_date": _git(["log", "-1", "--format=%ad", "--date=iso"], root),
        "commit_subject": _git(["log", "-1", "--format=%s"], root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "remote": _git(["remote", "get-url", "origin"], root),
        "dirty": bool(_git(["status", "--porcelain"], root)),
    }
    return info


def file_sha256(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    """计算文件 SHA-256（大文件分块读取）。"""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dependency_versions() -> Dict[str, Any]:
    """记录关键依赖版本。"""
    versions: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "mujoco", "gymnasium", "torch", "stable_baselines3", "sb3_contrib", "scipy"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = None
    return versions


def environment_vars() -> Dict[str, str]:
    keys = ("MUJOCO_GL", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS")
    return {k: os.environ.get(k, "") for k in keys}


def build_provenance(
    extra: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """汇总一次运行的 provenance。"""
    ckpt = Path(checkpoint_dir) if checkpoint_dir else paths.checkpoint_dir("LocomotionFull")
    model_path = ckpt / "checkpoint" / "best_model.zip"
    env_path = ckpt / "checkpoint" / "best_env.zip"
    cfg_path = ckpt / "locomotionFull.json"

    prov: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(paths.WORKSPACE_ROOT),
        "repos": [
            repo_info(paths.WORKSPACE_ROOT, "muscle-rl"),
            repo_info(paths.MSHUMAN_ROOT, "external/MS-Human-700"),
            repo_info(paths.MSGYM_ROOT, "external/msgym"),
        ],
        "checkpoint": {
            "dir": str(ckpt),
            "source": "https://github.com/LNSGroup/msgym/releases/download/Checkpoints/LocomotionFull.zip (tag: Checkpoints)",
            "model_zip": str(model_path),
            "model_zip_sha256": file_sha256(model_path),
            "env_zip": str(env_path),
            "env_zip_sha256": file_sha256(env_path),
            "config_json": str(cfg_path) if cfg_path.is_file() else None,
        },
        "model_xml": {
            "path": str(paths.MODEL_XML),
            "sha256": file_sha256(paths.MODEL_XML),
        },
        "dependencies": dependency_versions(),
        "env_vars": environment_vars(),
    }
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        prov["checkpoint"]["train_config"] = cfg
    if extra:
        prov["extra"] = extra
    return prov


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


__all__ = [
    "repo_info",
    "file_sha256",
    "dependency_versions",
    "build_provenance",
    "write_json",
]
