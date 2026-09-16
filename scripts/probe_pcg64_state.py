"""探测 best_env.zip 中 PCG64 / Generator 状态的 pickle 结构（numpy 2 格式）。"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hemirl import paths  # noqa: E402
from hemirl.numpy_compat import install_numpy2_pickle_compat  # noqa: E402

install_numpy2_pickle_compat()

EVENTS: list = []


class Recording:
    """记录 __setstate__ 收到的内容。"""

    def __init__(self, *a, **k):
        EVENTS.append(("init", self.__class__.__name__, a, k))

    def __setstate__(self, state):
        EVENTS.append(("setstate", self.__class__.__name__, state))
        self._state = state


class RecordingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            sub = module[len("numpy._core"):].lstrip(".")
            module = "numpy.core" + (f".{sub}" if sub else "")
        if module == "numpy.random._pcg64" and name == "PCG64":
            EVENTS.append(("resolve", "PCG64", module))
            return Recording
        if module == "numpy.random.bit_generator":
            EVENTS.append(("resolve-bg", name, module))
            return Recording
        if module == "numpy.random._pickle" and name.startswith("__"):
            EVENTS.append(("resolve-ctor", name, module))

            def ctor(*a, **k):
                EVENTS.append(("call-ctor", name, a, k))
                return Recording()

            return ctor
        return super().find_class(module, name)


def main() -> None:
    path = paths.checkpoint_dir("LocomotionFull") / "checkpoint" / "best_env.zip"
    try:
        with open(path, "rb") as fh:
            obj = RecordingUnpickler(fh).load()
        print("loaded ok:", type(obj))
    except Exception as exc:
        print("unpickle stopped with:", type(exc).__name__, exc)

    for ev in EVENTS:
        print(repr(ev)[:800])


if __name__ == "__main__":
    main()
