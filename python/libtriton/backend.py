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

from .transform import triton_graph_transform

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


class TritonGraphModule:
    """Wrapper around a transformed fx.GraphModule produced by triton_graph_backend."""

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.gm: torch.fx.GraphModule = triton_graph_transform(gm)
        self.fn: Optional[Callable[..., Any]] = None
        self._did_first_call: bool = False

    def __call__(self, *args: List[Any], **kwargs: Dict[Any, Any]) -> Any:
        with triton_kernel_scope(self.gm) as scope:
            if not self.fn:
                result = self.gm(*args, **kwargs)
                self.fn = self._build_fn(scope)
                return result
            else:
                return self.fn(*args, **kwargs)

    def _build_fn(self, scope: KernelScope) -> Callable[..., Any]:
        # TODO: this currently relies on the fact that the first call to the GraphModule will execute all kernels and thus trigger the hooks to capture the compiled kernels. We should ideally be able to capture the compiled kernels without relying on executing the GraphModule.
        for name, child in self.gm.named_children():
            if name.startswith("submod_torch_"):
                module = fx.stateless_fx_import(
                    child,
                    output_type=compiler_utils.OutputType.LINALG_ON_TENSORS,
                    model_name=name,
                )
                print(module)
            elif name.startswith("submod_triton_"):
                kernel: triton.compiler.CompiledKernel = scope.get_kernel(child)
                print(kernel.asm["ptx"])
            else:
                raise ValueError(f"unknown submodule type: {name}")
        return self.gm


class KernelScope:
    def __init__(self) -> None:
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
