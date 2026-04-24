from __future__ import annotations
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, TYPE_CHECKING

import torch
import tvm_ffi

if TYPE_CHECKING:
    import torch._higher_order_ops.triton_kernel_wrap
    import triton

from libtriton._C.libtriton_core import (
    capi_utils,
    compiler_utils,
    execution_engine,
    fx,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from .importers import TritonFxImporter
from .kernel import KernelBuilder

_CPU_EXECUTION_ENGINE_PIPELINE: Final[str] = (
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

_CUDA_EXECUTION_ENGINE_PIPELINE: Final[str] = (
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

_TORCH_TO_LINALG_NO_VERIFY_PIPELINE: Final[str] = (
    "builtin.module("
    "func.func(torch-restructure-non-constant-axes),"
    "func.func(torch-fuse-quantized-ops),"
    "func.func(convert-torch-to-tmtensor{allow-non-finites=true}),"
    "func.func(canonicalize),"
    "func.func(convert-torch-to-linalg{allow-non-finites=true}),"
    "func.func(canonicalize),"
    "func.func(convert-torch-to-scf),"
    "func.func(convert-torch-to-arith),"
    "func.func(convert-torch-to-tensor),"
    "convert-torch-conversion-to-mlprogram,"
    "func.func(memref-expand),"
    "func.func(canonicalize),"
    "func.func(resolve-shaped-type-result-dims),"
    "func.func(cse),"
    "torch-func-backend-type-conversion,"
    "func.func(canonicalize),"
    "func.func(torch-finalizing-backend-type-conversion)"
    ")"
)


class TritonGraphModule(object):
    """Compiles a torch.fx.GraphModule containing triton ops via TritonFxImporter."""

    def __init__(
        self, gm: torch.fx.GraphModule, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gm: torch.fx.GraphModule = gm
        self.fn: Optional[Callable[..., Any]] = None
        self.ctx: ir.Context = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        if not self.fn:
            self.gm(*args, **kwargs)
            self.fn = self._build_fn()
        return self.fn(*args, **kwargs)

    def _build_fn(self) -> Callable[..., Any]:
        importer = TritonFxImporter(context=self.ctx)
        # TODO: lower the resulting module and build an executable.
        # For now import to Torch IR, then try a custom backend pipeline
        # that mirrors torch-backend-to-linalg-on-tensors-backend-pipeline
        # without the final backend-contract verifier.
        module = fx.stateless_fx_import(
            self.gm,
            output_type=compiler_utils.OutputType.TORCH,
            model_name="main",
            fx_importer=importer,
        )
        # TODO: disable verifier for now
        # with self.ctx:
        #     passmanager.PassManager.parse(_TORCH_TO_LINALG_NO_VERIFY_PIPELINE).run(
        #         module.operation
        #     )
        print(module)
        return self.gm


class KernelScope(object):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)
        self._kernels: Dict[torch.fx.GraphModule, triton.JITFunction] = {}
        self._original_runs: List[Tuple[triton.JITFunction, Callable[..., Any]]] = []
        self._registry: Dict[torch.fx.GraphModule, triton.compiler.CompiledKernel] = {}
        self._active: bool = False

    def register_kernel(
        self, gm: torch.fx.GraphModule, kernel: triton.JITFunction
    ) -> None:
        self._kernels[gm] = kernel
        if self._active:
            self._patch_kernel_run(gm, kernel)

    def get_kernel(
        self, module: torch.fx.GraphModule
    ) -> Optional[triton.compiler.CompiledKernel]:
        return self._registry.get(module)

    def __enter__(self) -> KernelScope:
        self._active = True
        for gm, kernel in self._kernels.items():
            self._patch_kernel_run(gm, kernel)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for kernel, original_run in self._original_runs:
            kernel.run = original_run
        self._kernels = {}
        self._original_runs = []
        self._active = False

    def _patch_kernel_run(
        self, gm: torch.fx.GraphModule, kernel: triton.JITFunction
    ) -> None:
        original_run = kernel.run
        self._original_runs.append((kernel, original_run))
        kernel.run = self._build_kernel_run_hook(gm, original_run)

    def _build_kernel_run_hook(self, gm: torch.fx.GraphModule, fn: Any) -> Any:
        def run(*args: Any, **kwargs: Any) -> Any:
            compiled_kernel = fn(*args, **kwargs)
            self._registry[gm] = compiled_kernel
            return compiled_kernel

        return run


def triton_kernel_scope(gm: torch.fx.GraphModule) -> KernelScope:
    scope = KernelScope()
    for name, child in gm.named_children():
        if name.startswith("submod_triton_"):
            [triton_kernel_wrapper] = [
                node
                for node in child.graph.nodes
                if node.op == "call_function"
                and isinstance(
                    node.target,
                    torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional,
                )
            ]
            kernel = (
                torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
                    triton_kernel_wrapper.kwargs["kernel_idx"]
                )
            )
            scope.register_kernel(child, kernel)
    return scope


def triton_graph_backend(
    gm: torch.fx.GraphModule, example_inputs: List[Any]
) -> TritonGraphModule:
    return TritonGraphModule(gm)


def experimental_torch_mlir_execution_engine_backend(
    gm: torch.fx.GraphModule, example_inputs: List[Any]
) -> Any:
    """Temporary experimental backend, planned to be replaced by a full backend."""
    model_name = "main"
    use_cuda = any(
        isinstance(arg, torch.Tensor) and arg.device.type == "cuda"
        for arg in example_inputs
    )
    module = fx.export_and_import(
        gm,
        *example_inputs,
        output_type=compiler_utils.OutputType.LINALG_ON_TENSORS,
        func_name=model_name,
    )

    with module.context:
        register_all_dialects(module.context)
        register_all_passes()
        module.operation.regions[0].blocks[0].operations[0].operation.attributes[
            "tvm_ffi.emit_tvm_ffi_interface"
        ] = ir.UnitAttr.get()
        if use_cuda:
            major, minor = torch.cuda.get_device_capability()
            cuda_pipeline = _CUDA_EXECUTION_ENGINE_PIPELINE.format(
                chip=f"sm_{major}{minor}"
            )
            passmanager.PassManager.parse(cuda_pipeline).run(module.operation)
            engine = execution_engine.ExecutionEngine(
                module,
                shared_libs=[
                    capi_utils.find_capi_runtime_library("cuda"),
                    capi_utils.find_mlir_cuda_runtime_library(),
                ],
            )
        else:
            passmanager.PassManager.parse(_CPU_EXECUTION_ENGINE_PIPELINE).run(
                module.operation
            )
            engine = execution_engine.ExecutionEngine(
                module,
                shared_libs=[capi_utils.find_capi_runtime_library("cpu")],
            )
        engine.initialize()
        ptr = engine.raw_lookup(f"__tvm_ffi_{model_name}")
        if ptr is None:
            raise RuntimeError(f"symbol not found: __tvm_ffi_{model_name}")
        fn = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr, keep_alive_object=engine
        )

    def compiled(*args: Any) -> Tuple[Any, ...]:
        result = fn(*args)
        if isinstance(result, tuple):
            values = result
        elif isinstance(result, list):
            values = tuple(result)
        else:
            values = (result,)

        converted = tuple(
            torch.from_dlpack(value)
            if hasattr(value, "__dlpack__") and not isinstance(value, torch.Tensor)
            else value
            for value in values
        )
        return converted

    return compiled
