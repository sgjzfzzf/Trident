# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

import pathlib
import sysconfig
from typing import List

import torch
import tvm_ffi


def _trident_pkg_dir() -> pathlib.Path:
    """Return the installed ``trident`` package directory.

    In a regular install this is the same as ``parent of __file__``.
    In an editable install ``__file__`` points to the source tree while the
    compiled runtime libraries live under site-packages, so we fall back to
    the actual install location.
    """
    pkg_dir = pathlib.Path(__file__).resolve().parent
    if (pkg_dir / "core" / "_runtime_libs").is_dir():
        return pkg_dir
    # editable-install fallback
    site_packages = pathlib.Path(sysconfig.get_path("platlib"))
    return site_packages / "trident"


def find_capi_runtime_library() -> str:
    """Find the Trident Runtime library.

    Returns:
        Path to `libTridentCoreRuntime.so`

    Raises:
        RuntimeError: If the library is not found
    """
    pkg_dir = _trident_pkg_dir()
    capi_runtime_lib = pkg_dir / "core" / "_runtime_libs" / "libTridentCoreRuntime.so"
    assert capi_runtime_lib.is_file(), (
        f"missing Trident Runtime library: {capi_runtime_lib}"
    )
    return f"{capi_runtime_lib}"


def find_ffi_exception_library() -> str:
    """Find the Trident FFI library.

    Returns:
        Path to ``libTridentFFI.so``

    Raises:
        RuntimeError: If the library is not found
    """
    # ffi library is in trident/ffi/_runtime_libs/
    pkg_dir = _trident_pkg_dir()
    ffi_exception_lib = pkg_dir / "ffi" / "_runtime_libs" / "libTridentFFI.so"
    assert ffi_exception_lib.is_file(), (
        f"missing Trident FFI Exception library: {ffi_exception_lib}"
    )
    return f"{ffi_exception_lib}"


def find_mlir_cuda_runtime_library() -> str:
    """Find the MLIR CUDA runtime library.

    Returns:
        Path to `libmlir_cuda_runtime.so`

    Raises:
        RuntimeError: If the library is not found
    """
    pkg_dir = _trident_pkg_dir()
    cuda_runtime_lib = pkg_dir / "core" / "_runtime_libs" / "libmlir_cuda_runtime.so"
    assert cuda_runtime_lib.is_file(), (
        f"missing MLIR CUDA runtime library: {cuda_runtime_lib}"
    )
    return f"{cuda_runtime_lib}"


def find_runtime_libraries() -> List[str]:
    """Find the Trident runtime libraries.

    Returns:
        Paths to the Trident runtime libraries

    Raises:
        RuntimeError: If any runtime library is not found
    """
    return [
        find_capi_runtime_library(),
        find_ffi_exception_library(),
        find_mlir_cuda_runtime_library(),
        torch._C.__file__,
        tvm_ffi.libinfo.find_libtvm_ffi(),
    ]
