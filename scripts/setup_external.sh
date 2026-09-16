#!/usr/bin/env bash
#
# 获取上游只读依赖到 external/，并 pin 到与本次实验完全一致的 commit。
#
#   bash scripts/setup_external.sh
#
# 上游仓库（只读，不要修改）：
#   external/MS-Human-700   LNSGroup/MS-Human-700   官方肌骨模型 XML + 资产
#   external/msgym          LNSGroup/msgym          官方 Gymnasium 环境 + DynSyn-SAC 脚本
#
# 本工作区不把上游源码纳入自己的 git 仓库（见 .gitignore），
# 用本脚本按 commit 复原，保证可复现。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/external"

MSHUMAN_NAME="MS-Human-700"
MSHUMAN_URL="https://github.com/LNSGroup/MS-Human-700.git"
MSHUMAN_SHA="2d686957aefd5739cf4d2859a6acfd94c8f84400"
MSHUMAN_XML_SHA256="755124766e0e4ed9ddac60c8c4f505f94a6adaa2d51665108eee0cb1f677681d"

MSGYM_NAME="msgym"
MSGYM_URL="https://github.com/LNSGroup/msgym.git"
MSGYM_SHA="ad3aac166dab3577d7d16b4a03faccef94c7aac7"

fetch_pinned() {
  local name="$1" url="$2" sha="$3" dir="$4"
  if [ -d "$dir/.git" ]; then
    echo "[skip] $name 已存在: $dir"
  else
    echo "[clone] $name -> $dir"
    mkdir -p "$dir"
    git -C "$dir" init -q
    if git -C "$dir" remote get-url origin >/dev/null 2>&1; then
      git -C "$dir" remote set-url origin "$url"
    else
      git -C "$dir" remote add origin "$url"
    fi
    # GitHub 支持按 commit sha 浅取；失败则回退完整克隆
    if git -C "$dir" fetch -q --depth 1 origin "$sha" 2>/dev/null; then
      git -C "$dir" checkout -q FETCH_HEAD
    else
      echo "      按 sha 浅取失败，回退为完整抓取…"
      git -C "$dir" fetch -q origin
      git -C "$dir" checkout -q "$sha"
    fi
  fi

  local got
  got="$(git -C "$dir" rev-parse HEAD)"
  if [ "$got" != "$sha" ]; then
    echo "[error] $name 的 HEAD 是 $got，期望 $sha" >&2
    echo "        如需强制对齐：git -C \"$dir\" fetch origin && git -C \"$dir\" checkout $sha" >&2
    exit 1
  fi
  echo "[ok]    $name @ $got"
}

fetch_pinned "$MSHUMAN_NAME" "$MSHUMAN_URL" "$MSHUMAN_SHA" "$EXT/$MSHUMAN_NAME"
fetch_pinned "$MSGYM_NAME" "$MSGYM_URL" "$MSGYM_SHA" "$EXT/$MSGYM_NAME"

# 上游 msgym 通过 git submodule 把模型放在 msgym/MS-Human-700。
# 本工作区用符号链接指向独立克隆（hemirl/paths.py: ensure_msgym_model_link 也会自动处理）。
LINK="$EXT/msgym/msgym/MS-Human-700"
if [ -d "$LINK" ] && [ ! -L "$LINK" ] && [ -z "$(ls -A "$LINK" 2>/dev/null)" ]; then
  rmdir "$LINK"   # 上游 submodule 未初始化留下的空目录
fi
if [ ! -e "$LINK" ]; then
  ln -s "$EXT/$MSHUMAN_NAME" "$LINK"
  echo "[link]  $LINK -> $EXT/$MSHUMAN_NAME"
else
  echo "[skip]  $LINK 已存在"
fi

echo
echo "模型 XML : $EXT/$MSHUMAN_NAME/MS-Human-700.xml"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$EXT/$MSHUMAN_NAME/MS-Human-700.xml" | awk '{print $1}')"
  if [ "$actual" = "$MSHUMAN_XML_SHA256" ]; then
    echo "sha256  : $actual  (与实验一致)"
  else
    echo "sha256  : $actual"
    echo "          ⚠ 与实验记录不一致，期望 $MSHUMAN_XML_SHA256"
  fi
fi

cat <<'EOF'

下一步：
  1) 下载官方 checkpoint（约 214 MB，不入库）：
       mkdir -p artifacts/checkpoints && cd artifacts/checkpoints
       curl -L -o LocomotionFull.zip \
         https://github.com/LNSGroup/msgym/releases/download/Checkpoints/LocomotionFull.zip
       unzip -q LocomotionFull.zip && cd ../..
  2) 自检：
       MUJOCO_GL=egl PYTHONPATH=. python scripts/run_tests.py --with-heavy
EOF
