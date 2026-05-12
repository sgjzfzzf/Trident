from __future__ import annotations
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
)

import torch
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
from libtriton._C.libtriton_core.dialects import arith, func, llvm
from .guards import GuardParser, Guards
from .importer import TritonFxImporter


@tvm_ffi.register_error("GuardMatchError")
class GuardMatchError(RuntimeError):
    pass


class TritonGraphModule(object):
    """Compiles a torch.fx.GraphModule containing triton ops via TritonFxImporter."""

    _parser: GuardParser = GuardParser()

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Callable[..., Any]] = fn
        self.executor: Optional[Callable[..., Any]] = None
        self.guard_to_fx_map: Dict[Guards, ir.Module] = {}
        self.ctx: ir.Context = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()

    @property
    def _kind_value(self) -> str:
        return "GuardMatchError\x00"

    @property
    def _message_value(self) -> str:
        return "No suitable guard matched\x00"

    @property
    def _ptr_type(self) -> ir.Type:
        return ir.Type.parse("!llvm.ptr")

    @property
    def _i32_type(self) -> ir.Type:
        return ir.IntegerType.get_signless(32)

    @property
    def _kind_str_ty(self) -> ir.Type:
        return ir.Type.parse(f"!llvm.array<{len(self._kind_value)} x i8>")

    @property
    def _message_str_ty(self) -> ir.Type:
        return ir.Type.parse(f"!llvm.array<{len(self._message_value)} x i8>")

    @property
    def _func_type(self) -> ir.Type:
        return ir.FunctionType.get(
            [self._ptr_type, self._ptr_type, self._i32_type, self._ptr_type],
            [self._i32_type],
        )

    @property
    def _func_name(self) -> str:
        return f"__tvm_ffi_{self.fn.__name__}"

    @property
    def _kind_global_name(self) -> str:
        return "__libtriton_guard_match_error_kind"

    @property
    def _message_global_name(self) -> str:
        return "__libtriton_no_suitable_guard_matched"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.executor:
            return self.executor(*args, **kwargs)
        else:
            return self.compile(*args, **kwargs)

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        ret: Any = self.fn(*args, **kwargs)
        if self.executor is None:
            gm, guards = torch._dynamo.export(
                self.fn, aten_graph=True, assume_static_by_default=True
            )(*args, **kwargs)
            parsed_guards_set: Guards = TritonGraphModule._parser.parse_guards(guards)
            # TODO: Build the actual IR module from the fx.GraphModule once the importer supports all necessary ops.
            self.guard_to_fx_map[parsed_guards_set] = self._build_ir_module(gm)
            module: ir.Module = self._build_stub_ir_module()
            self.executor = TritonGraphModule._build_execution_engine_callable(
                module,
                model_name=self.fn.__name__,
            )
        return ret

    def _build_ir_module(self, gm: torch.fx.GraphModule) -> ir.Module:
        importer: TritonFxImporter = TritonFxImporter(context=self.ctx)
        module: ir.Module = fx.stateless_fx_import(
            gm,
            output_type=compiler_utils.OutputType.TORCH,
            model_name=self.fn.__name__,
            fx_importer=importer,
        )
        with self.ctx:
            TritonGraphModule._mark_for_tvm_ffi_interface(module, self.fn.__name__)
            major, minor = torch.cuda.get_device_capability()
            pipeline = TritonGraphModule._build_pipeline(chip=f"sm_{major}{minor}")
            passmanager.PassManager.parse(pipeline).run(module.operation)
        return module

    def _build_stub_ir_module(self) -> ir.Module:
        """Build a stub IR module with a TVM-FFI compatible wrapper that returns -1.

        This creates an IR module containing a single wrapper function with the TVM-FFI
        calling convention using the tvm_ffi.any type from the TVM-FFI Dialect.
        The function simply returns -1 and does not process any actual computation.
        """
        with self.ctx, ir.Location.unknown():
            module: ir.Module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                llvm.GlobalOp(
                    self._kind_str_ty,
                    self._kind_global_name,
                    ir.Attribute.parse("#llvm.linkage<internal>"),
                    constant=True,
                    value=ir.StringAttr.get(self._kind_value),
                )
                llvm.GlobalOp(
                    self._message_str_ty,
                    self._message_global_name,
                    ir.Attribute.parse("#llvm.linkage<internal>"),
                    constant=True,
                    value=ir.StringAttr.get(self._message_value),
                )
                func_op: Any = func.FuncOp(self._func_name, self._func_type)
                entry_block: ir.Block = ir.Block.create_at_start(
                    func_op.body,
                    [self._ptr_type, self._ptr_type, self._i32_type, self._ptr_type],
                )
                with ir.InsertionPoint(entry_block):
                    kind_ptr: ir.Value = llvm.AddressOfOp(
                        self._ptr_type, self._kind_global_name
                    ).result
                    message_ptr: ir.Value = llvm.AddressOfOp(
                        self._ptr_type, self._message_global_name
                    ).result
                    ir.Operation.create(
                        "tvm_ffi.error_set_raised_from_c_str",
                        operands=[kind_ptr, message_ptr],
                    )
                    const_minus_one: ir.Value = arith.constant(self._i32_type, -1)
                    func.return_([const_minus_one])
            major, minor = torch.cuda.get_device_capability()
            pipeline = TritonGraphModule._build_pipeline(chip=f"sm_{major}{minor}")
            passmanager.PassManager.parse(pipeline).run(module.operation)
            return module

    @staticmethod
    def _build_execution_engine_callable(
        module: ir.Module, model_name: str
    ) -> Callable[..., Any]:
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

        def f(*args: Any) -> Any:
            result = fn(*args)
            if isinstance(result, (list, tuple)):
                return tuple(
                    TritonGraphModule._canonicalize_ret(value) for value in result
                )
            else:
                return TritonGraphModule._canonicalize_ret(result)

        return f

    @staticmethod
    def _build_pipeline(chip: str) -> str:
        passes: List[str] = [
            "func.func(torch-restructure-non-constant-axes)",
            "func.func(torch-fuse-quantized-ops)",
            "func.func(convert-torch-to-tmtensor{allow-non-finites=true})",
            "func.func(convert-torch-to-linalg{allow-non-finites=true})",
            "func.func(convert-torch-to-scf)",
            "func.func(convert-torch-to-arith)",
            "func.func(convert-torch-to-tensor)",
            "convert-torch-conversion-to-mlprogram",
            "torch-func-backend-type-conversion",
            "func.func(torchext-normalize-operands)",
            "func.func(torch-finalizing-backend-type-conversion)",
            "one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map}",
            "emit-tvm-ffi-interface",
            "convert-linalg-to-parallel-loops",
            "func.func(gpu-map-parallel-loops)",
            "convert-parallel-loops-to-gpu",
            "gpu-kernel-outlining",
            "finalize-memref-to-llvm{use-generic-functions=1}",
            f"nvvm-attach-target{{O=3 chip={chip}}}",
            "gpu.module(convert-gpu-to-nvvm{index-bitwidth=64})",
            "convert-arith-to-llvm",
            "convert-torchext-to-llvm",
            "gpu-to-llvm",
            "torchext-async-kernel-launch",
            "convert-torchext-to-llvm",
            "gpu-module-to-binary{format=fatbin}",
            "convert-func-to-llvm",
            "func.func(canonicalize)",
            "func.func(cse)",
            "reconcile-unrealized-casts",
        ]
        return "builtin.module({})".format(", ".join(passes))

    @staticmethod
    def _canonicalize_ret(val: Any) -> Any:
        return (
            torch.from_dlpack(val)
            if hasattr(val, "__dlpack__") and not isinstance(val, torch.Tensor)
            else val
        )

    @staticmethod
    def _mark_for_tvm_ffi_interface(module: ir.Module, model_name: str) -> None:
        for op in module.body.operations:
            if (
                "sym_name" in op.attributes
                and ir.StringAttr(op.attributes["sym_name"]).value == model_name
            ):
                op.attributes["tvm_ffi.emit_tvm_ffi_interface"] = ir.UnitAttr.get()
