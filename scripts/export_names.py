"""导出模型命名清单：actuator / body / joint 名称，用于设计肌群映射。"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hemirl import paths  # noqa: E402


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(paths.MODEL_XML))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]

    out = Path("reports/model_names.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"actuators": names, "bodies": bodies, "joints": joints}, indent=1),
        encoding="utf-8",
    )

    # 名称后缀模式
    suff = Counter()
    for n in names:
        mm = re.search(r"_(r|l|R|L)$", n)
        suff[mm.group(1) if mm else "none"] += 1
    print("actuator side suffix counts:", dict(suff))

    print("\n--- bodies (all) ---")
    for i, b in enumerate(bodies):
        print(f"{i:4d} {b}")

    print("\n--- joint name suffix counts ---")
    js = Counter()
    for n in joints:
        mm = re.search(r"_(r|l|R|L)$", n)
        js[mm.group(1) if mm else "none"] += 1
    print(dict(js))
    print("\n--- first 40 joints ---")
    for i, j in enumerate(joints[:40]):
        print(f"{i:4d} {j}")


if __name__ == "__main__":
    main()
