import pathlib
from typing import List

import torch


def find_capi_runtime_library() -> str:
    """Find the LibTriton Runtime library.

    Returns:
        Path to `libLibTritonCoreRuntime.so`

    Raises:
        RuntimeError: If the library is not found
    """
    module_dir = pathlib.Path(__file__).resolve().parent
    runtime_lib = module_dir / "_runtime_libs" / "libLibTritonCoreRuntime.so"
    if not runtime_lib.is_file():
        raise RuntimeError(f"missing LibTriton Runtime library: {runtime_lib}")
    return str(runtime_lib)


def find_mlir_cuda_runtime_library() -> str:
    """Find the MLIR CUDA runtime library.

    Returns:
        Path to `libmlir_cuda_runtime.so`

    Raises:
        RuntimeError: If the library is not found
    """
    module_dir = pathlib.Path(__file__).resolve().parent
    cuda_runtime_lib = module_dir / "_runtime_libs" / "libmlir_cuda_runtime.so"
    if not cuda_runtime_lib.is_file():
        raise RuntimeError(f"missing MLIR CUDA runtime library: {cuda_runtime_lib}")
    return str(cuda_runtime_lib)


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
    ]
