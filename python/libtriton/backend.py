from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import torch
import torch._dynamo.backends.common
import tvm_ffi

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
from .importer import TritonFxImporter

if TYPE_CHECKING:
    import triton

_PIPELINE: Final[str] = (
    "builtin.module("
    "func.func(torch-restructure-non-constant-axes),"
    "func.func(torch-fuse-quantized-ops),"
    "func.func(convert-torch-to-tmtensor{{allow-non-finites=true}}),"
    "func.func(convert-torch-to-linalg{{allow-non-finites=true}}),"
    "func.func(convert-torch-to-scf),"
    "func.func(convert-torch-to-arith),"
    "func.func(convert-torch-to-tensor),"
    "convert-torch-conversion-to-mlprogram,"
    "torch-func-backend-type-conversion,"
    "func.func(torchext-normalize-operands),"
    "func.func(torch-finalizing-backend-type-conversion),"
    "one-shot-bufferize{{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map}},"
    "emit-tvm-ffi-interface,"
    "convert-torchext-to-llvm,"
    "convert-linalg-to-parallel-loops,"
    "func.func(gpu-map-parallel-loops),"
    "convert-parallel-loops-to-gpu,"
    "gpu-kernel-outlining,"
    "torchext-async-kernel-launch,"
    "convert-arith-to-llvm,"
    "finalize-memref-to-llvm{{use-generic-functions=1}},"
    "nvvm-attach-target{{O=3 chip={chip}}},"
    "gpu.module(convert-gpu-to-nvvm{{index-bitwidth=64}}),"
    "gpu-to-llvm,"
    "gpu-module-to-binary{{format=fatbin}},"
    "convert-func-to-llvm,"
    "func.func(canonicalize),"
    "func.func(cse),"
    "reconcile-unrealized-casts"
    ")"
)


class TritonGraphModule(object):
    """Compiles a torch.fx.GraphModule containing triton ops via TritonFxImporter."""

    def __init__(
        self, gm: torch.fx.GraphModule, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gm: Final[torch.fx.GraphModule] = gm
        self.fn: Optional[Callable[..., Any]] = None
        self.ctx: ir.Context = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        if self.fn:
            return self.fn(*args, **kwargs)
        else:
            ret: triton.KernelInterface = self.gm(*args, **kwargs)
            self.fn = self._build_fn(*args)
            return ret

    def _build_fn(self, *example_inputs: Any) -> Callable[..., Any]:
        model_name = "main"
        importer = TritonFxImporter(context=self.ctx)
        module: ir.Module = fx.stateless_fx_import(
            self.gm,
            output_type=compiler_utils.OutputType.TORCH,
            model_name=model_name,
            fx_importer=importer,
        )
        with self.ctx:
            TritonGraphModule._mark_for_tvm_ffi_interface(module, model_name)
            major, minor = torch.cuda.get_device_capability()
            pipeline = _PIPELINE.format(chip=f"sm_{major}{minor}")
            passmanager.PassManager.parse(pipeline).run(module.operation)
            print(module)
            return TritonGraphModule._build_execution_engine_callable(
                module,
                model_name=model_name,
            )

    @staticmethod
    def _mark_for_tvm_ffi_interface(
        module: ir.Module, model_name: str = "main"
    ) -> None:
        for op in module.body.operations:
            if (
                "sym_name" in op.attributes
                and ir.StringAttr(op.attributes["sym_name"]).value == model_name
            ):
                op.attributes["tvm_ffi.emit_tvm_ffi_interface"] = ir.UnitAttr.get()

    @staticmethod
    def _build_execution_engine_callable(
        module: ir.Module, model_name: str
    ) -> Callable[[Sequence[Any]], Tuple[Any, ...]]:
        engine: execution_engine.ExecutionEngine = execution_engine.ExecutionEngine(
            module,
            shared_libs=capi_utils.find_runtime_libraries(),
        )

        engine.initialize()
        ptr: Callable[..., Any] = engine.raw_lookup(f"__tvm_ffi_{model_name}")
        assert ptr is not None, f"symbol not found: __tvm_ffi_{model_name}"
        fn: Callable[..., Any] = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr, keep_alive_object=engine
        )

        def f(*args: Sequence[Any]) -> Tuple[Any, ...]:
            result = fn(*args)
            if isinstance(result, tuple):
                values = result
            elif isinstance(result, list):
                values = tuple(result)
            else:
                values = (result,)

            return tuple(
                torch.from_dlpack(value)
                if hasattr(value, "__dlpack__") and not isinstance(value, torch.Tensor)
                else value
                for value in values
            )

        return f


def triton_graph_backend(
    gm: torch.fx.GraphModule, example_inputs: List[Any]
) -> Callable[..., Any]:
    return torch._dynamo.backends.common.aot_autograd(
        fw_compiler=triton_fw_compiler_impl
    )(gm, example_inputs)


def triton_fw_compiler_impl(
    gm: torch.fx.GraphModule, example_inputs: List[Any]
) -> TritonGraphModule:
    return TritonGraphModule(gm)
