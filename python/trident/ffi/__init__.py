# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Trident FFI runtime — Exception types and FFI utilities."""

import ctypes
import pathlib
from typing import Any

import tvm_ffi


def _load_ffi_library() -> None:
    """Preload libTridentFFI.so so that tvm_ffi can discover its registered types."""
    import sysconfig

    module_dir = pathlib.Path(__file__).resolve().parent
    lib_path = module_dir / "_runtime_libs" / "libTridentFFI.so"

    if not lib_path.is_file():
        # editable install: the .so lives under site-packages while __init__.py
        # is loaded from the source tree.  Fall back to the installed location.
        site_packages = pathlib.Path(sysconfig.get_path("platlib"))
        lib_path = (
            site_packages / "trident" / "ffi" / "_runtime_libs" / "libTridentFFI.so"
        )

    if lib_path.is_file():
        ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


_load_ffi_library()


def Exception(kind: str) -> Any:
    """Construct a ``trident.ffi.Exception`` FFI object from *kind*.

    Args:
        kind: A human-readable tag describing the exception (e.g.
            ``"GuardMatchException"``).

    Returns:
        An opaque tvm::ffi ObjectRef that can be interrogated through the
        tvm_ffi runtime type system.
    """
    return tvm_ffi.get_global_func("trident.ffi.Exception")(kind)


def get_exception_index() -> int:
    """Return the runtime type index reserved for ``trident.ffi.Exception``."""
    return tvm_ffi.get_global_func("trident.ffi.GetExceptionIndex")()
