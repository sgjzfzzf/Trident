import pathlib
from typing import Final

import numpy as np
import tvm_ffi

import libtriton_core


MLIR_FILE: Final[pathlib.Path] = (
    pathlib.Path(__file__).parent / "Inputs" / "tvm_ffi_tensor_add.mlir"
)
TEST_FUNCTION: Final[str] = "tensor_add_kernel"


def _lower_to_llvm_dialect() -> libtriton_core.ir.Module:
    ctx = libtriton_core.ir.Context()
    with ctx:
        libtriton_core.register_all_dialects(ctx)
        libtriton_core.register_all_passes()
        module = libtriton_core.ir.Module.parse(MLIR_FILE.read_text(encoding="utf-8"))
        pipeline = (
            "builtin.module("
            "emit-tvm-ffi-interface,"
            "convert-to-llvm,"
            "convert-index-to-llvm,"
            "convert-arith-to-llvm,"
            "finalize-memref-to-llvm,"
            "convert-func-to-llvm,"
            "reconcile-unrealized-casts"
            ")"
        )
        libtriton_core.passmanager.PassManager.parse(pipeline).run(module.operation)
        return module


def _build_tvm_ffi_function(module: libtriton_core.ir.Module) -> tvm_ffi.Function:
    engine = libtriton_core.execution_engine.ExecutionEngine(
        module,
        shared_libs=[
            libtriton_core.capi_utils.find_capi_runtime_library(libtriton_core.__file__)
        ],
    )
    function_ptr = engine.raw_lookup(f"__tvm_ffi_{TEST_FUNCTION}")
    if function_ptr is None:
        raise RuntimeError(f"function pointer not found: __tvm_ffi_{TEST_FUNCTION}")
    return tvm_ffi.Function.__from_mlir_packed_safe_call__(
        function_ptr,
        keep_alive_object=engine,
    )


def main() -> None:
    module = _lower_to_llvm_dialect()
    function = _build_tvm_ffi_function(module)

    lhs = np.array([1, 2, 3, 4], dtype=np.int64)
    rhs = np.array([10, 20, 30, 40], dtype=np.int64)
    out = function(lhs, rhs)
    out_np = np.from_dlpack(out)
    np.testing.assert_array_equal(out_np, lhs + rhs)


if __name__ == "__main__":
    main()
