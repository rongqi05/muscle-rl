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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# ------------------------------------------------------------------ 代码版本


def dirty_files(root: Optional[Path] = None) -> List[str]:
    """工作区未提交改动的文件清单（``git status --porcelain`` 的路径部分）。"""
    root = Path(root) if root else paths.WORKSPACE_ROOT
    out = _git(["status", "--porcelain"], root)
    if not out:
        return []
    files: List[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:  # 重命名：取目标路径
            path = path.split(" -> ")[-1]
        files.append(path)
    return files


def dirty_patch_sha256(
    root: Optional[Path] = None,
    sub_paths: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """未提交改动的补丁哈希（默认 ``git diff HEAD``，可限定路径）。

    用于在「工作区 dirty」时仍然能标识**代码快照**：同一 commit + 同一补丁哈希
    才代表同一份代码。

    注意参数名用 ``sub_paths`` 而非 ``paths``：后者与模块级的 ``hemirl.paths`` 同名，
    会遮蔽掉默认值里的 ``paths.WORKSPACE_ROOT``。
    """
    root = Path(root) if root else paths.WORKSPACE_ROOT
    cmd = ["git", "--no-pager", "diff", "HEAD"]
    if sub_paths:
        cmd += ["--", *sub_paths]
    try:
        out = subprocess.run(cmd, cwd=str(root), capture_output=True, timeout=60)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout).hexdigest()


#: 参与「代码快照」判定的路径前缀：只有它们变脏才改变 ``describe``。
#: 运行输出（``reports/`` / ``runs/``）会在实验过程中被自己改写，
#: 若把它们算作脏，每个结果都会标上 ``+dirty`` 而失去标识意义。
CODE_PATHS: Tuple[str, ...] = ("hemirl", "scripts", "tests", "configs")


def code_version(root: Optional[Path] = None) -> Dict[str, Any]:
    """本项目代码版本：commit + **代码**脏状态 + 代码补丁哈希 + 输出脏文件。

    ``describe`` 是可直接写进结果表的短标识；只有 ``hemirl/`` / ``scripts/`` /
    ``tests/`` / ``configs/`` 下的改动才会让后缀出现，形如
    ``ab452c4+code-dirty:1f3c9a7e``。

    字段说明：

    * ``code_dirty_files`` / ``patch_sha256``：决定代码快照标识；
    * ``other_dirty_files``（如 ``reports/`` / ``runs/`` / ``README.md``）：
      记录但不影响 ``describe``；
    * ``patch_sha256_all``：全量补丁哈希，供需要完整 diff 的场景使用。
    """
    root = Path(root) if root else paths.WORKSPACE_ROOT
    info = repo_info(root, "muscle-rl")
    files = dirty_files(root)
    code_files = [f for f in files if f.split("/", 1)[0] in CODE_PATHS]
    other_files = [f for f in files if f.split("/", 1)[0] not in CODE_PATHS]
    patch_code = dirty_patch_sha256(root, CODE_PATHS) if code_files else None
    patch_all = dirty_patch_sha256(root) if files else None
    untracked = sorted(f for f in files if _git(["ls-files", "--error-unmatch", f], root) is None)
    commit = info.get("commit") or "unknown"
    short = commit[:7] if commit != "unknown" else "unknown"
    tag = f"+code-dirty:{patch_code[:8]}" if code_files and patch_code else ("+code-dirty" if code_files else "")
    return {
        "commit": commit,
        "commit_subject": info.get("commit_subject"),
        "branch": info.get("branch"),
        "remote": info.get("remote"),
        "dirty": bool(files),
        "code_dirty": bool(code_files),
        "code_dirty_files": code_files,
        "other_dirty_files": other_files,
        "untracked_files": untracked,
        "patch_sha256": patch_code,
        "patch_sha256_all": patch_all,
        "code_paths_used_for_describe": list(CODE_PATHS),
        "describe": f"{short}{tag}",
    }


def upstream_dirty(root: Path) -> Dict[str, Any]:
    """上游仓库的脏文件列表，并区分「模型符号链接变化」与「源码变化」。

    本工作区会在 ``msgym`` 内建立 ``msgym/MS-Human-700`` 符号链接（上游 submodule 未
    初始化时的空目录会被替换），这属于**非源码**变化，必须与真正的源码改动区分开，
    否则「上游只读」这一约定无法核对。
    """
    root = Path(root)
    if not (root / ".git").exists():
        return {"path": str(root), "is_git": False, "dirty": False}
    raw = _git(["status", "--porcelain"], root) or ""
    symlink_changes: List[str] = []
    source_changes: List[str] = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[-1]
        full = root / path
        if full.is_symlink() or path.endswith("MS-Human-700"):
            symlink_changes.append(path)
        else:
            source_changes.append(path)
    return {
        "path": str(root),
        "is_git": True,
        "dirty": bool(raw.splitlines()),
        "dirty_raw": raw.splitlines(),
        "symlink_changes": symlink_changes,
        "source_changes": source_changes,
        "n_source_changes": len(source_changes),
    }


def model_file_info(path: Path) -> Dict[str, Any]:
    """实际加载的模型文件：路径、是否为符号链接、解析后的真实路径与哈希。"""
    path = Path(path)
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
    }
    if path.is_symlink():
        info["symlink_target"] = os.readlink(path)
    if path.exists():
        info["resolved"] = str(path.resolve())
        info["sha256"] = file_sha256(path)
        info["size_bytes"] = path.stat().st_size
    return info


def run_provenance(
    *,
    entry: str,
    args: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[Path] = None,
    loaded_model_path: Optional[Path] = None,
    strength: Optional[Dict[str, Any]] = None,
    policy_inference: Optional[Dict[str, Any]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """一次运行的完整 provenance（覆盖任务书第七节的全部字段）。"""
    prov = build_provenance(extra=extra, checkpoint_dir=checkpoint_dir)
    prov["entry"] = entry
    prov["args"] = args
    prov["code_version"] = code_version()
    prov["upstream"] = {
        "MS-Human-700": {**repo_info(paths.MSHUMAN_ROOT, "MS-Human-700"), **upstream_dirty(paths.MSHUMAN_ROOT)},
        "msgym": {**repo_info(paths.MSGYM_ROOT, "msgym"), **upstream_dirty(paths.MSGYM_ROOT)},
    }
    if loaded_model_path is not None:
        prov["loaded_model_file"] = model_file_info(loaded_model_path)
    if strength is not None:
        prov["strength"] = strength
    if policy_inference is not None:
        prov["policy_inference"] = policy_inference
    if seeds is not None:
        prov["seeds"] = list(seeds)
    return prov


__all__ = [
    "repo_info",
    "file_sha256",
    "dependency_versions",
    "build_provenance",
    "write_json",
    "code_version",
    "dirty_files",
    "dirty_patch_sha256",
    "upstream_dirty",
    "model_file_info",
    "run_provenance",
]
