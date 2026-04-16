import unittest
from typing import Final

import numpy as np
import torch
import tvm_ffi

from libtriton._C.libtriton_core import (
    capi_utils,
    execution_engine,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from libtriton._C.libtriton_core.dialects import func, linalg, tensor

_TEST_FUNCTION: Final[str] = "tensor_add_kernel"
_CPU_PIPELINE: Final[str] = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},"
    "emit-tvm-ffi-interface,"
    "convert-linalg-to-loops,"
    "convert-scf-to-cf,"
    "convert-to-llvm,"
    "convert-arith-to-llvm,"
    "convert-cf-to-llvm,"
    "finalize-memref-to-llvm{use-generic-functions=1},"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)

_CUDA_PIPELINE: Final[str] = (
    "builtin.module("
    "one-shot-bufferize{{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map}},"
    "emit-tvm-ffi-interface,"
    "convert-linalg-to-parallel-loops,"
    "func.func(gpu-map-parallel-loops),"
    "convert-parallel-loops-to-gpu,"
    "gpu-kernel-outlining,"
    "lower-affine,"
    "convert-arith-to-llvm,"
    "convert-index-to-llvm,"
    "finalize-memref-to-llvm{{use-generic-functions=1}},"
    "nvvm-attach-target{{O=3 chip={chip}}},"
    "gpu.module(convert-gpu-to-nvvm{{index-bitwidth=64}}),"
    "gpu-to-llvm,"
    "gpu-module-to-binary{{format=fatbin}},"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)


def _build_module(ctx: ir.Context) -> ir.Module:
    """Build tensor add with pure tensor+linalg IR."""
    with ir.Location.unknown():
        i64 = ir.IntegerType.get_signless(64)
        tensor4xi64 = ir.RankedTensorType.get([4], i64)
        add_fn_type = ir.FunctionType.get([tensor4xi64, tensor4xi64], [tensor4xi64])
        module = ir.Module.create()

        # func.func @tensor_add_kernel(%x, %y) -> tensor<4xi64>
        #     attributes {tvm_ffi.emit_tvm_ffi_interface}
        kernel_fn = func.FuncOp("tensor_add_kernel", add_fn_type)
        kernel_fn.operation.attributes["tvm_ffi.emit_tvm_ffi_interface"] = (
            ir.UnitAttr.get()
        )
        module.body.append(kernel_fn.operation)
        entry = ir.Block.create_at_start(
            kernel_fn.regions[0], [tensor4xi64, tensor4xi64]
        )
        with ir.InsertionPoint(entry):
            init = tensor.empty([4], i64)
            out = linalg.add(entry.arguments[0], entry.arguments[1], outs=[init])
            func.ReturnOp([out])

    return module


def _compile(runtime_type: str) -> tvm_ffi.Function:
    ctx = ir.Context()
    with ctx:
        register_all_dialects(ctx)
        register_all_passes()
        module = _build_module(ctx)
        passmanager.PassManager.parse(_CPU_PIPELINE).run(module.operation)
        engine = execution_engine.ExecutionEngine(
            module,
            shared_libs=[capi_utils.find_capi_runtime_library(runtime_type)],
        )
        engine.initialize()
        ptr = engine.raw_lookup(f"__tvm_ffi_{_TEST_FUNCTION}")
        if ptr is None:
            raise RuntimeError(f"symbol not found: __tvm_ffi_{_TEST_FUNCTION}")
        return tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr, keep_alive_object=engine
        )


def _compile_cuda() -> tvm_ffi.Function:
    major, minor = torch.cuda.get_device_capability()
    pipeline = _CUDA_PIPELINE.format(chip=f"sm_{major}{minor}")
    ctx = ir.Context()
    with ctx:
        register_all_dialects(ctx)
        register_all_passes()
        module = _build_module(ctx)
        passmanager.PassManager.parse(pipeline).run(module.operation)
        engine = execution_engine.ExecutionEngine(
            module,
            shared_libs=[
                capi_utils.find_capi_runtime_library("cuda"),
                capi_utils.find_mlir_cuda_runtime_library(),
            ],
        )
        engine.initialize()
        ptr = engine.raw_lookup(f"__tvm_ffi_{_TEST_FUNCTION}")
        if ptr is None:
            raise RuntimeError(f"symbol not found: __tvm_ffi_{_TEST_FUNCTION}")
        return tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr, keep_alive_object=engine
        )


class TestTVMFFIAddNumpy(unittest.TestCase):
    def test_add(self) -> None:
        fn = _compile("cpu")
        lhs = np.array([1, 2, 3, 4], dtype=np.int64)
        rhs = np.array([10, 20, 30, 40], dtype=np.int64)
        out = np.from_dlpack(fn(lhs, rhs))
        np.testing.assert_array_equal(out, lhs + rhs)


class TestTVMFFIAddTorchCPU(unittest.TestCase):
    def test_add(self) -> None:
        fn = _compile("cpu")
        lhs = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        rhs = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        out = torch.from_dlpack(fn(lhs, rhs))
        torch.testing.assert_close(out, lhs + rhs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTVMFFIAddTorchCUDA(unittest.TestCase):
    def test_add(self) -> None:
        fn = _compile_cuda()
        lhs = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
        rhs = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="cuda")
        out = torch.from_dlpack(fn(lhs, rhs))
        torch.testing.assert_close(out.cpu(), (lhs + rhs).cpu())


if __name__ == "__main__":
    unittest.main()
