import unittest
from typing import Final, Tuple

import torch
import triton
import triton.language as tl

import libtriton.kernel
from libtriton._C.libtriton_core import (
    capi_utils,
    execution_engine,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)  # type: ignore[import-not-found]

_BLOCK_SIZE: Final[int] = 1024
_GPU_BINARY_TO_LLVM_PIPELINE: Final[str] = (
    "builtin.module("
    "convert-arith-to-llvm,"
    "convert-index-to-llvm,"
    "convert-func-to-llvm,"
    "gpu-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTritonAddKernelCubin(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda")
        cls.n_elements = 4096
        cls.x = torch.rand(cls.n_elements, device=cls.device)
        cls.y = torch.rand(cls.n_elements, device=cls.device)
        cls.output = torch.empty_like(cls.x)

        def grid(meta) -> Tuple[int]:
            return (triton.cdiv(cls.n_elements, meta["BLOCK_SIZE"]),)

        cls.launch_grid = grid({"BLOCK_SIZE": _BLOCK_SIZE})

        cls.compiled_kernel = add_kernel[grid](
            cls.x, cls.y, cls.output, cls.n_elements, BLOCK_SIZE=_BLOCK_SIZE
        )
        cls.kernel_builder = libtriton.kernel.KernelBuilder(cls.compiled_kernel)

    def test_capture_cubin_from_add_kernel(self) -> None:
        self.assertIsNotNone(self.kernel_builder.cubin)
        self.assertIsInstance(self.kernel_builder.cubin, bytes)
        self.assertGreater(len(self.kernel_builder.cubin), 0)
        self.assertEqual(self.kernel_builder.cubin[:4], b"\x7fELF")

    def test_build_mlir_gpu_launch_module_from_cubin(self) -> None:
        ctx = ir.Context()
        register_all_dialects(ctx)
        module = self.kernel_builder.build_gpu_launch_module(
            ctx,
            self.launch_grid,
            fn_name="launch_add_kernel",
        )
        module_text = f"{module}"
        self.assertIn("gpu.launch_func", module_text)
        self.assertIn("gpu.binary", module_text)
        self.assertEqual(
            self.kernel_builder.parameters,
            [
                ("x_ptr", "*fp32"),
                ("y_ptr", "*fp32"),
                ("output_ptr", "*fp32"),
                ("n_elements", "i32"),
                ("BLOCK_SIZE", "constexpr"),
            ],
        )
        self.assertIn("!llvm.ptr", module_text)
        self.assertIn("i32", module_text)
        self.assertIn(
            f"@{self.kernel_builder.kernel_name}::@{self.kernel_builder.kernel_name}",
            module_text,
        )

    def test_lower_gpu_launch_module_to_llvm_dialect(self) -> None:
        ctx = ir.Context()
        register_all_dialects(ctx)
        module = self.kernel_builder.build_gpu_launch_module(
            ctx,
            self.launch_grid,
            fn_name="launch_add_kernel",
        )
        with ctx:
            register_all_passes()
            passmanager.PassManager.parse(_GPU_BINARY_TO_LLVM_PIPELINE).run(
                module.operation
            )

        shared_libs = [
            capi_utils.find_capi_runtime_library("cuda"),
            capi_utils.find_mlir_cuda_runtime_library(),
        ]
        engine = execution_engine.ExecutionEngine(module, shared_libs=shared_libs)
        engine.initialize()
        fn_ptr = engine.raw_lookup("launch_add_kernel")
        self.assertIsNotNone(fn_ptr)


if __name__ == "__main__":
    unittest.main()
