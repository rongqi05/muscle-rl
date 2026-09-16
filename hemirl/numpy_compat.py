"""numpy 2.x pickle 兼容 shim。

**问题**：官方 checkpoint 由 numpy 2.4 序列化，pickle 中记录了
``numpy._core.multiarray`` / ``numpy._core.numeric`` 等模块路径。
本项目使用的 conda 环境是 numpy 1.26，其中的对应模块名为 ``numpy.core.*``，
因此 ``VecNormalize.load`` 与 SB3 的 ``load_from_zip_file`` 都会抛
``ModuleNotFoundError: No module named 'numpy._core'``。

**为什么不升级 numpy**：``env_isaaclab`` 同时服务于 Isaac Lab 技术栈，
升级 numpy 到 2.x 会带来不必要的连锁影响。这里采用最小侵入的兼容层。

**做法**：若当前 numpy 是 1.x，则把 ``numpy.core`` 及其子模块注册为
``numpy._core`` 的别名。numpy 1.26 的 ``numpy.core.numeric._frombuffer``
与 numpy 2.x 的 ``numpy._core.numeric._frombuffer`` 语义一致，可安全互换。
"""

from __future__ import annotations

import importlib
import sys
from typing import List

#: numpy 2.x 中被移动/重命名的子模块（对应 numpy 1.x 的 numpy.core.*）
_CORE_SUBMODULES: List[str] = [
    "multiarray",
    "numeric",
    "umath",
    "_multiarray_umath",
    "_multiarray_tests",
    "fromnumeric",
    "_exceptions",
    "shape_base",
    "_methods",
    "numerictypes",
    "memmap",
    "records",
    "function_base",
    "getlimits",
    "einsumfunc",
    "_type_aliases",
    "arrayprint",
    "cversions",
    "defchararray",
    "overrides",
    "_asarray",
    "_dtype",
    "_dtype_ctypes",
    "_functions",
    "_ufunc_config",
    "_umath_tests",
]

_INSTALLED = False


def install_numpy2_pickle_compat(verbose: bool = False) -> dict:
    """安装 numpy 2.x → 1.x 的 pickle 兼容层（幂等）。

    做三件事（仅当 numpy 主版本 < 2 时）:

    1. 注册 ``numpy._core`` → ``numpy.core`` 模块别名；
    2. 用容错版本替换 ``numpy.random._pickle`` 中的
       ``__bit_generator_ctor`` / ``__generator_ctor`` / ``__randomstate_ctor``；
    3. 用容错子类替换私有模块里的 BitGenerator 类
       （``numpy.random._pcg64.PCG64`` 等），使其 ``__setstate__`` 同时接受
       numpy 2 的 ``(state_dict, seed_sequence)`` 与 numpy 1 的 ``state_dict``。

    第 3 步是必要的：SB3 的 ``load_from_zip_file`` 通过 cloudpickle 反序列化内嵌
    载荷，走的是标准 pickle 通道，无法被自定义 Unpickler 覆盖。
    """
    global _INSTALLED
    import numpy

    report = {
        "numpy_version": numpy.__version__,
        "already_installed": _INSTALLED,
        "needed": False,
        "aliased": [],
        "skipped": [],
        "patched_ctors": [],
        "patched_bit_generators": [],
    }
    if _INSTALLED:
        report["needed"] = True
        return report
    if int(numpy.__version__.split(".")[0]) >= 2:
        report["note"] = "numpy >= 2，无需 shim"
        _INSTALLED = True
        return report

    report["needed"] = True
    try:
        import numpy.core as core  # type: ignore
    except Exception as exc:  # pragma: no cover
        report["error"] = repr(exc)
        return report

    sys.modules.setdefault("numpy._core", core)
    report["aliased"].append("numpy._core")
    for name in _CORE_SUBMODULES:
        target = f"numpy.core.{name}"
        alias = f"numpy._core.{name}"
        if alias in sys.modules:
            continue
        try:
            mod = importlib.import_module(target)
        except Exception:
            report["skipped"].append(alias)
            continue
        sys.modules[alias] = mod
        report["aliased"].append(alias)

    # --- 2) 随机数 pickle 构造函数 ---
    try:
        import numpy.random._pickle as npr_pickle

        for fn_name, replacement in (
            ("__bit_generator_ctor", _tolerant_bit_generator_ctor),
            ("__generator_ctor", _tolerant_generator_ctor),
            ("__randomstate_ctor", _tolerant_randomstate_ctor),
        ):
            original = getattr(npr_pickle, fn_name, None)
            if original is None or getattr(original, "_hemirl_compat", False):
                continue
            replacement._hemirl_compat = True  # type: ignore[attr-defined]
            setattr(npr_pickle, fn_name, replacement)
            report["patched_ctors"].append(f"numpy.random._pickle.{fn_name}")
    except Exception as exc:
        report["ctor_error"] = repr(exc)

    # --- 3) BitGenerator.__setstate__ 兼容 ---
    for name in _BIT_GENERATOR_NAMES:
        compat = _compat_bit_generator_class(name)
        if compat is None:
            continue
        for module_name in _BIT_GENERATOR_MODULES.get(name, ()):
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                continue
            current = getattr(mod, name, None)
            if current is compat:
                continue
            if current is None:
                continue
            setattr(mod, name, compat)
            report["patched_bit_generators"].append(f"{module_name}.{name}")

    _INSTALLED = True
    if verbose:
        print(
            f"[numpy_compat] aliased={len(report['aliased'])} "
            f"ctors={report['patched_ctors']} bitgens={report['patched_bit_generators']}"
        )
    return report


# ------------------------------------------------------------------ 随机数 pickle 兼容


def _tolerant_bit_generator_ctor(*args):
    """接受「类」或「名字字符串」两种入参的 BitGenerator 构造器。

    numpy 1.26 的 ``__bit_generator_ctor`` 只接受字符串名字；numpy 2.x 的 pickle
    会传入 *类对象*。这里两种都接受。
    """
    import numpy as np

    registry = {
        "MT19937": np.random.MT19937,
        "PCG64": np.random.PCG64,
        "PCG64DXSM": np.random.PCG64DXSM,
        "Philox": np.random.Philox,
        "SFC64": np.random.SFC64,
    }
    for a in args:
        if isinstance(a, str) and a in registry:
            return registry[a]()
        if isinstance(a, type) and issubclass(a, np.random.BitGenerator):
            return a()
    raise ValueError(f"无法从参数 {args!r} 构造 BitGenerator")


def _tolerant_generator_ctor(*args):
    """构造 ``Generator``。

    numpy 2.x 的 pickle 直接传入**已构造好的 BitGenerator 实例**；
    numpy 1.26 的签名是 ``(bit_generator_name, bit_generator_ctor)``。两种都支持。
    """
    import numpy as np

    for a in args:
        if isinstance(a, np.random.BitGenerator):
            return np.random.Generator(a)
    return np.random.Generator(_tolerant_bit_generator_ctor(*args))


def _tolerant_randomstate_ctor(*args):
    import numpy as np

    for a in args:
        if isinstance(a, np.random.BitGenerator):
            return np.random.RandomState(a)
    return np.random.RandomState(_tolerant_bit_generator_ctor(*args))


#: numpy 2.x 在 pickle 中会把 BitGenerator 的 state 序列化为 (state_dict, SeedSequence)
_BIT_GENERATOR_NAMES = ("MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64")

#: BitGenerator 类在 numpy 私有模块中的位置（pickle 里记录的模块路径）。
#: 只替换私有模块属性，不碰 `numpy.random.PCG64` 这类公开名字，把影响范围降到最小。
_BIT_GENERATOR_MODULES = {
    "MT19937": ("numpy.random._mt19937",),
    "PCG64": ("numpy.random._pcg64",),
    "PCG64DXSM": ("numpy.random._pcg64",),
    "Philox": ("numpy.random._philox",),
    "SFC64": ("numpy.random._sfc64",),
}
_COMPAT_BG_CACHE: dict = {}


def _compat_bit_generator_class(name: str):
    """返回一个能同时接受 numpy 1.x / 2.x ``__setstate__`` 格式的 BitGenerator 子类。"""
    import numpy as np

    if name in _COMPAT_BG_CACHE:
        return _COMPAT_BG_CACHE[name]
    real = getattr(np.random, name, None)
    if not (isinstance(real, type) and issubclass(real, np.random.BitGenerator)):
        return None

    class _CompatBitGenerator(real):  # type: ignore[misc,valid-type]
        """numpy 2 的 state 是 ``(state_dict, seed_sequence)``；numpy 1.x 只接受 ``state_dict``。"""

        def __setstate__(self, state):
            if isinstance(state, tuple) and state and isinstance(state[0], dict):
                state = state[0]
            return super().__setstate__(state)

    _CompatBitGenerator.__name__ = name
    _CompatBitGenerator.__qualname__ = name
    _COMPAT_BG_CACHE[name] = _CompatBitGenerator
    return _CompatBitGenerator


def _running_numpy_major() -> int:
    import numpy

    return int(numpy.__version__.split(".")[0])


def _compat_unpickler_class():
    import pickle

    class _NumpyCompatUnpickler(pickle.Unpickler):
        """把 numpy 2.x 序列化的全局引用映射到当前 numpy 版本。"""

        _RANDOM_CTORS = {
            "__bit_generator_ctor": _tolerant_bit_generator_ctor,
            "__generator_ctor": _tolerant_generator_ctor,
            "__randomstate_ctor": _tolerant_randomstate_ctor,
        }

        def find_class(self, module: str, name: str):
            if _running_numpy_major() < 2:
                # numpy 2 把 numpy.core 改名为 numpy._core
                if module.startswith("numpy._core"):
                    sub = module[len("numpy._core"):].lstrip(".")
                    module = "numpy.core" + (f".{sub}" if sub else "")
                # 随机数 pickle 辅助函数：签名在 1.x / 2.x 之间变了
                if module == "numpy.random._pickle" and name in self._RANDOM_CTORS:
                    return self._RANDOM_CTORS[name]
                # BitGenerator 的 __setstate__ 入参格式变了
                if module.startswith("numpy.random") and name in _BIT_GENERATOR_NAMES:
                    compat = _compat_bit_generator_class(name)
                    if compat is not None:
                        return compat
            return super().find_class(module, name)

    return _NumpyCompatUnpickler


def compat_pickle_load(path) -> object:
    """用兼容 unpickler 读取文件（numpy 2 生成的 pickle 在 numpy 1.x 下也可读）。"""
    import pickle

    cls = _compat_unpickler_class()
    with open(path, "rb") as fh:
        return cls(fh).load()


def compat_pickle_loads(payload: bytes) -> object:
    import io
    import pickle

    cls = _compat_unpickler_class()
    return cls(io.BytesIO(payload)).load()


__all__ = [
    "install_numpy2_pickle_compat",
    "compat_pickle_load",
    "compat_pickle_loads",
]
