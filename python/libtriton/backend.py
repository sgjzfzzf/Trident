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
from libtriton._C.libtriton_core.dialects import (
    arith,
    bufferization,
    func,
    llvm,
    scf,
    transform,
    _dlpack_ops_gen as dlpack_dialect,
    _tvm_ffi_ops_gen as tvm_ffi_dialect,
)
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
        self.guard_to_model_name: Dict[Guards, str] = {}
        self.ctx: ir.Context = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()
        self.importer: TritonFxImporter = TritonFxImporter(context=self.ctx)

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
    def _i64_type(self) -> ir.Type:
        return ir.IntegerType.get_signless(64)

    @property
    def _any_type(self) -> ir.Type:
        return ir.Type.parse("!tvm_ffi.any")

    @property
    def _any_llvm_type(self) -> ir.Type:
        return ir.Type.parse("!llvm.struct<(i32, i32, i64)>")

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
            gm, gs = torch._dynamo.export(
                self.fn, aten_graph=True, assume_static_by_default=True
            )(*args, **kwargs)
            guards: Guards = TritonGraphModule._parser.parse_guards(gs)
            model_name: str = f"{self.fn.__name__}_{hash(guards)}"
            module: ir.Module = fx.stateless_fx_import(
                gm,
                output_type=compiler_utils.OutputType.TORCH,
                model_name=model_name,
                fx_importer=self.importer,
            )
            with self.ctx:
                passmanager.PassManager.parse(
                    TritonGraphModule._build_torch_pipeline()
                ).run(module.operation)
            self.guard_to_fx_map[guards] = module
            self.guard_to_model_name[guards] = model_name
            module = self._build_ir_module()
            self.executor = TritonGraphModule._build_execution_engine_callable(
                module,
                model_name=self.fn.__name__,
            )
        return ret

    @staticmethod
    def _lookup_func_type(module: ir.Module, sym_name: str) -> ir.FunctionType:
        for op in module.body.operations:
            if "sym_name" in op.attributes and "function_type" in op.attributes:
                value: str = ir.StringAttr(op.attributes["sym_name"]).value
                if value == sym_name:
                    type_attr: ir.TypeAttr = ir.TypeAttr(op.attributes["function_type"])
                    return ir.FunctionType(type_attr.value)
        raise RuntimeError(f"cannot find function symbol: {sym_name}")

    @staticmethod
    def _is_tensor_type(target_ty: ir.Type) -> bool:
        return str(target_ty).startswith("tensor<")

    @staticmethod
    def _is_memref_type(target_ty: ir.Type) -> bool:
        return str(target_ty).startswith("memref<")

    @staticmethod
    def _tensor_to_memref_type(tensor_ty: ir.Type) -> ir.Type:
        tensor_str: str = str(tensor_ty)
        if not tensor_str.startswith("tensor<"):
            raise RuntimeError(f"not a tensor type: {tensor_str}")
        return ir.Type.parse(f"memref<{tensor_str[len('tensor<') :]}")

    def _emit_unbox_any_arg(
        self,
        packed_args_ptr: ir.Value,
        index: int,
        target_ty: ir.Type,
    ) -> ir.Value:
        index_value: ir.Value = arith.constant(self._i64_type, index)
        arg_slot_ptr: ir.Value = llvm.GEPOp(
            self._ptr_type,
            packed_args_ptr,
            [index_value],
            ir.DenseI32ArrayAttr.get([-2147483648]),
            self._any_llvm_type,
            None,
        ).result
        any_value_llvm: ir.Value = llvm.LoadOp(self._any_llvm_type, arg_slot_ptr).result
        any_value: ir.Value = tvm_ffi_dialect.AnyFromLLVMOp(
            self._any_type, any_value_llvm
        ).output
        if TritonGraphModule._is_tensor_type(target_ty):
            dl_tensor_ty: ir.Type = ir.Type.parse("!dlpack.tensor")
            dl_tensor: ir.Value = tvm_ffi_dialect.ToOp(dl_tensor_ty, any_value).output
            memref_ty: ir.Type = TritonGraphModule._tensor_to_memref_type(target_ty)
            memref: ir.Value = dlpack_dialect.ToMemRefOp(memref_ty, dl_tensor).output
            return bufferization.ToTensorOp(
                target_ty,
                memref,
                restrict=True,
                writable=True,
            ).result
        elif TritonGraphModule._is_memref_type(target_ty):
            dl_tensor_ty = ir.Type.parse("!dlpack.tensor")
            dl_tensor = tvm_ffi_dialect.ToOp(dl_tensor_ty, any_value).output
            return dlpack_dialect.ToMemRefOp(target_ty, dl_tensor).output
        else:
            return tvm_ffi_dialect.ToOp(target_ty, any_value).output

    def _emit_box_return(self, packed_result_ptr: ir.Value, value: ir.Value) -> None:
        value_type: ir.Type = value.type
        managed_ty: ir.Type = ir.Type.parse("!dlpack.managed_tensor")
        object_handle_ty: ir.Type = ir.Type.parse("!tvm_ffi.object_handle")
        zero: ir.Value = arith.constant(self._i32_type, 0)
        if TritonGraphModule._is_tensor_type(value_type):
            memref_ty: ir.Type = TritonGraphModule._tensor_to_memref_type(value_type)
            memref: ir.Value = bufferization.ToBufferOp(memref_ty, value).buffer
            managed: ir.Value = dlpack_dialect.FromMemRefOwnedOp(
                managed_ty,
                memref,
            ).output
            handle: ir.Value = tvm_ffi_dialect.TensorFromDLPackOp(
                object_handle_ty,
                managed,
                zero,
                zero,
            ).output
            boxed: ir.Value = tvm_ffi_dialect.ToOp(self._any_type, handle).output
        elif TritonGraphModule._is_memref_type(value_type):
            managed: ir.Value = dlpack_dialect.FromMemRefOwnedOp(
                managed_ty,
                value,
            ).output
            handle: ir.Value = tvm_ffi_dialect.TensorFromDLPackOp(
                object_handle_ty,
                managed,
                zero,
                zero,
            ).output
            boxed = tvm_ffi_dialect.ToOp(self._any_type, handle).output
        else:
            boxed = tvm_ffi_dialect.ToOp(self._any_type, value).output
        boxed_llvm: ir.Value = tvm_ffi_dialect.AnyToLLVMOp(
            self._any_llvm_type, boxed
        ).output
        llvm.StoreOp(boxed_llvm, packed_result_ptr)

    def _emit_dispatch_call(
        self,
        packed_args_ptr: ir.Value,
        packed_result_ptr: ir.Value,
        callee_name: str,
        callee_ty: ir.FunctionType,
    ) -> ir.Value:
        call_args: List[ir.Value] = [
            self._emit_unbox_any_arg(packed_args_ptr, idx, target_ty)
            for idx, target_ty in enumerate(callee_ty.inputs)
        ]
        result_types: List[ir.Type] = list(callee_ty.results)
        call_op: Any = func.CallOp(result_types, callee_name, call_args)
        [result] = call_op.results
        self._emit_box_return(packed_result_ptr, result)
        return arith.constant(self._i32_type, 0)

    def _emit_fallback_error(self) -> ir.Value:
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
        return arith.constant(self._i32_type, -1)

    def _emit_dispatch_if_chain(
        self,
        dispatch_items: List[tuple[Guards, str, ir.FunctionType]],
        index: int,
        packed_args_ptr: ir.Value,
        packed_result_ptr: ir.Value,
    ) -> ir.Value:
        if index >= len(dispatch_items):
            return self._emit_fallback_error()
        guard, callee_name, callee_ty = dispatch_items[index]
        cond: ir.Value = guard.build_ir(context=self.ctx)
        if_op: Any = scf.IfOp(cond, [self._i32_type], has_else=True)
        with ir.InsertionPoint(if_op.then_block):
            success: ir.Value = self._emit_dispatch_call(
                packed_args_ptr,
                packed_result_ptr,
                callee_name,
                callee_ty,
            )
            scf.YieldOp([success])
        with ir.InsertionPoint(if_op.else_block):
            fallback: ir.Value = self._emit_dispatch_if_chain(
                dispatch_items,
                index + 1,
                packed_args_ptr,
                packed_result_ptr,
            )
            scf.YieldOp([fallback])
        [result] = if_op.results
        return result

    def _build_ir_module(self) -> ir.Module:
        """Build an IR module with a guard-dispatch TVM-FFI wrapper.

        The wrapper iterates all compiled guard variants and emits a conditional
        dispatch chain. On the first matched guard it unboxes packed
        tvm_ffi.any arguments into the callee parameter types, invokes the
        corresponding submodule main function, boxes the optional single return
        value back to tvm_ffi.any, and returns success (0). If no guard matches,
        it sets the GuardMatchError and returns -1.
        """
        with self.ctx, ir.Location.unknown():
            module: ir.Module = ir.Module.create()
            module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
            for imported_module in self.guard_to_fx_map.values():
                transform.interpreter.copy_symbols_and_merge_into(
                    module, imported_module
                )
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
                packed_args_ptr: ir.Value = entry_block.arguments[1]
                packed_result_ptr: ir.Value = entry_block.arguments[3]
                dispatch_items: List[tuple[Guards, str, ir.FunctionType]] = [
                    (
                        guards,
                        self.guard_to_model_name[guards],
                        TritonGraphModule._lookup_func_type(
                            module, self.guard_to_model_name[guards]
                        ),
                    )
                    for guards in self.guard_to_fx_map
                ]

                with ir.InsertionPoint(entry_block):
                    status: ir.Value = self._emit_dispatch_if_chain(
                        dispatch_items,
                        0,
                        packed_args_ptr,
                        packed_result_ptr,
                    )
                    func.return_([status])
            major, minor = torch.cuda.get_device_capability()
            pipeline: str = TritonGraphModule._build_builtin_pipeline(
                chip=f"sm_{major}{minor}"
            )
            passmanager.PassManager.parse(pipeline).run(module.operation)
            return module

    @staticmethod
    def _build_execution_engine_callable(
        module: ir.Module,
        model_name: str,
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
    def _build_builtin_pipeline(chip: str) -> str:
        passes: List[str] = [
            "one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map}",
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
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "convert-func-to-llvm",
            "func.func(canonicalize)",
            "func.func(cse)",
            "reconcile-unrealized-casts",
        ]
        return "builtin.module({})".format(", ".join(passes))

    def _build_torch_pipeline() -> str:
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
