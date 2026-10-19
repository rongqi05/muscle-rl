"""把周期检查点目录转成评估/诊断所需的目录形状。

周期检查点保存为 ``runs/<tag>/checkpoints/ckpt_N/{ckpt_model.zip,ckpt_env.zip,...}``，
而评估/诊断入口（``hemirl.policy.load_sb3_stack``、``OfficialEnvConfig.from_checkpoint``）
要求 ``<dir>/checkpoint/best_model.zip`` + ``checkpoint/best_env.zip`` + 根目录
``locomotionFull.json``。本脚本在暂存目录里用符号链接搭出这个形状（不复制、不改
上游检查点）。

用法::

    PYTHONPATH=. python scripts/stage_checkpoint_for_eval.py \
        --src runs/train/healthy_v6/checkpoints/ckpt_100008 \
        --dst runs/eval_staging/v6_100k \
        [--env-json runs/train/healthy_v1/locomotionFull.json]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="周期检查点目录（含 ckpt_model.zip / ckpt_env.zip）")
    parser.add_argument("--dst", required=True, help="暂存输出目录")
    parser.add_argument("--env-json", default=None,
                        help="locomotionFull.json 来源（默认 src 的父父目录（run 根）下查找）")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    ckpt = dst / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)

    model = src / "ckpt_model.zip"
    env_zip = src / "ckpt_env.zip"
    if not model.is_file():
        raise SystemExit(f"缺模型文件: {model}")
    if not env_zip.is_file():
        raise SystemExit(f"缺归一化文件: {env_zip}")

    (ckpt / "best_model.zip").symlink_to(model.resolve())
    (ckpt / "best_env.zip").symlink_to(env_zip.resolve())

    env_json = Path(args.env_json) if args.env_json else None
    if env_json is None:
        for cand in (src.parent.parent / "locomotionFull.json",
                     src.parent / "locomotionFull.json",
                     paths.RUNS_ROOT / "train" / "healthy_v1" / "locomotionFull.json"):
            if cand.is_file():
                env_json = cand
                break
    if env_json is None or not env_json.is_file():
        raise SystemExit("找不到 locomotionFull.json，请用 --env-json 指定")
    dst_json = dst / "locomotionFull.json"
    if not dst_json.exists():
        shutil.copy2(env_json, dst_json)

    print(f"[staged] {dst}")
    print(f"  best_model.zip -> {model.resolve()}")
    print(f"  best_env.zip   -> {env_zip.resolve()}")
    print(f"  locomotionFull.json <- {env_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
