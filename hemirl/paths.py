"""路径解析：本工作区的模型、上游源码、checkpoint 与输出目录。

所有路径集中在此处，避免散落的相对路径假设。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

EXTERNAL_ROOT = WORKSPACE_ROOT / "external"
MSHUMAN_ROOT = EXTERNAL_ROOT / "MS-Human-700"
MSGYM_ROOT = EXTERNAL_ROOT / "msgym"

MODEL_XML = MSHUMAN_ROOT / "MS-Human-700.xml"
MODEL_LOCOMOTION_XML = MSHUMAN_ROOT / "MS-Human-700-Locomotion.xml"
MODEL_MANIPULATION_XML = MSHUMAN_ROOT / "MS-Human-700-Manipulation.xml"

ARTIFACTS_ROOT = WORKSPACE_ROOT / "artifacts"
CHECKPOINT_ROOT = ARTIFACTS_ROOT / "checkpoints"
RUNS_ROOT = WORKSPACE_ROOT / "runs"
REPORTS_ROOT = WORKSPACE_ROOT / "reports"
CONFIGS_ROOT = WORKSPACE_ROOT / "configs"


def ensure_msgym_on_path() -> str:
    """把上游 `external/msgym` 加入 `sys.path`，返回其绝对路径。

    msgym 未安装为 site-package（上游使用 uv），因此这里显式注入。必须在
    `import msgym` 之前调用。
    """
    import sys

    path = str(MSGYM_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def msgym_model_dir() -> Path:
    """msgym 包内期望的模型目录 `msgym/MS-Human-700`。

    上游通过 git submodule 填充该目录。这里若为空，则由
    `ensure_msgym_model_link()` 建立到 `external/MS-Human-700` 的符号链接。
    """
    return MSGYM_ROOT / "msgym" / "MS-Human-700"


def ensure_msgym_model_link() -> Path:
    """保证 `msgym/MS-Human-700` 指向独立克隆的官方模型。

    上游 `msgym/envs/utils.py:get_ms_human_model_path` 会依次查找
    `<pkg>/MS-Human-700/<file>` 与 `<repo_root>/MS-Human-700/<file>`。
    我们用符号链接满足第一条，避免重复下载模型资产。

    注意：git submodule 未初始化时该路径会是一个**空目录**，此函数会将其替换为符号链接；
    若目录非空（例如用户已正确初始化 submodule），则原样保留。
    """
    link = msgym_model_dir()
    if link.is_symlink():
        return link
    if link.exists():
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()  # 上游 submodule 未初始化留下的空目录
        else:
            return link
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(MSHUMAN_ROOT, link, target_is_directory=True)
    return link


def ensure_dynsyn_scripts_on_path() -> str:
    """把上游 `DynSyn-SAC/SB3-Scripts` 加入 `sys.path`（用于 `wrapper` 与 `DynSyn` 导入）。"""
    import sys

    path = str(MSGYM_ROOT / "DynSyn-SAC" / "SB3-Scripts")
    if path not in sys.path:
        sys.path.append(path)
    return path


def ensure_dynsyn_package_on_path() -> str:
    """把上游 `DynSyn-SAC`（含 `DynSyn` 包）加入 `sys.path`。"""
    import sys

    path = str(MSGYM_ROOT / "DynSyn-SAC")
    if path not in sys.path:
        sys.path.append(path)
    return path


def checkpoint_dir(name: str = "LocomotionFull") -> Path:
    """官方 checkpoint 解压目录。"""
    return CHECKPOINT_ROOT / name


def run_dir(tag: str, root: Optional[Path] = None) -> Path:
    """实验输出目录（不存在则创建）。"""
    root = RUNS_ROOT if root is None else Path(root)
    path = root / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(filename: str = "MS-Human-700.xml") -> Path:
    """按文件名返回官方模型 XML 路径。"""
    path = MSHUMAN_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    return path
