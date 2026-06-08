import pathlib
from typing import List

import torch
import tvm_ffi


def find_capi_runtime_library() -> str:
    """Find the LibTriton Runtime library.

    Returns:
        Path to `libLibTritonCoreRuntime.so`

    Raises:
        RuntimeError: If the library is not found
    """
    module_dir = pathlib.Path(__file__).resolve().parent
    capi_runtime_lib = module_dir / "_runtime_libs" / "libLibTritonCoreRuntime.so"
    assert capi_runtime_lib.is_file(), (
        f"missing LibTriton Runtime library: {capi_runtime_lib}"
    )
    return f"{capi_runtime_lib}"


def find_mlir_cuda_runtime_library() -> str:
    """Find the MLIR CUDA runtime library.

    Returns:
        Path to `libmlir_cuda_runtime.so`

    Raises:
        RuntimeError: If the library is not found
    """
    module_dir = pathlib.Path(__file__).resolve().parent
    cuda_runtime_lib = module_dir / "_runtime_libs" / "libmlir_cuda_runtime.so"
    assert cuda_runtime_lib.is_file(), (
        f"missing MLIR CUDA runtime library: {cuda_runtime_lib}"
    )
    return f"{cuda_runtime_lib}"


def find_runtime_libraries() -> List[str]:
    """Find the LibTriton runtime libraries.

    Returns:
        Paths to the LibTriton runtime libraries

    Raises:
        RuntimeError: If any runtime library is not found
    """
    return [
        find_capi_runtime_library(),
        find_mlir_cuda_runtime_library(),
        torch._C.__file__,
        tvm_ffi.libinfo.find_libtvm_ffi(),
    ]
