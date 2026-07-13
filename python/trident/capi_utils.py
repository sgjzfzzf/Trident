# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

import functools
import importlib.metadata as im
import pathlib
from typing import List

import torch
import tvm_ffi


@functools.cache
def _trident_pkg_dir() -> pathlib.Path:
    """Return the installed ``trident`` package directory.

    Uses ``importlib.metadata`` to locate the package root via the
    distribution RECORD — the same approach :mod:`tvm_ffi.libinfo`
    uses to find its own shared libraries.
    """
    return (im.distribution("trident")._path.parent / "trident").resolve()


def find_capi_runtime_library() -> str:
    """Find the Trident Runtime library.

    Returns:
        Path to `libTridentCoreRuntime.so`

    Raises:
        RuntimeError: If the library is not found
    """
    capi_runtime_lib: pathlib.Path = (
        _trident_pkg_dir() / "lib" / "libTridentCoreRuntime.so"
    )
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
    ffi_exception_lib: pathlib.Path = _trident_pkg_dir() / "lib" / "libTridentFFI.so"
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
    cuda_runtime_lib: pathlib.Path = (
        _trident_pkg_dir() / "lib" / "libmlir_cuda_runtime.so"
    )
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
