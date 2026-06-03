from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
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
    dlpack,
    func,
    llvm,
    scf,
    transform,
    tvm_ffi as tvm_ffi_d,
)
from .error import GuardMatchException
from .guards import Guards, parse_guards
from .importer import LibTritonFxImporter


@dataclass(frozen=True)
class GuardVariant(object):
    guard: Guards
    module: ir.Module
    wrapper_name: str
    wrapper_type: ir.FunctionType


class LibTritonGraphModule(object):
    """Compiles a torch.fx.GraphModule containing triton ops via LibTritonFxImporter."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Callable[..., Any]] = fn
        self.guard_variants: List[GuardVariant] = []
        self.ctx: ir.Context = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()
        self.executor: Callable[..., Any] = self.stub_compile()

    @property
    def _dl_tensor_type(self) -> ir.Type:
        return ir.Type.parse("!dlpack.tensor")

    @property
    def _object_handle_type(self) -> ir.Type:
        return ir.Type.parse("!tvm_ffi.object_handle")

    @property
    def _any_llvm_type(self) -> ir.Type:
        return llvm.StructType.get_literal(
            [self._i32_type, self._i32_type, self._i64_type]
        )

    @property
    def _any_type(self) -> ir.Type:
        return ir.Type.parse("!tvm_ffi.any")

    @property
    def _func_name(self) -> str:
        return f"__tvm_ffi_{self.fn.__name__}"

    @property
    def _func_type(self) -> ir.Type:
        return ir.FunctionType.get(
            [self._ptr_type, self._ptr_type, self._i32_type, self._ptr_type],
            [self._i32_type],
        )

    @property
    def _i32_type(self) -> ir.Type:
        return ir.IntegerType.get_signless(32)

    @property
    def _i64_type(self) -> ir.Type:
        return ir.IntegerType.get_signless(64)

    @property
    def _kind_global_name(self) -> str:
        return "__libtriton_guard_match_error_kind"

    @property
    def _kind_str_ty(self) -> ir.Type:
        return ir.Type.parse(f"!llvm.array<{len(self._kind_value)} x i8>")

    @property
    def _kind_value(self) -> str:
        return "GuardMatchException\x00"

    @property
    def _message_global_name(self) -> str:
        return "__libtriton_no_suitable_guard_matched"

    @property
    def _message_str_ty(self) -> ir.Type:
        return ir.Type.parse(f"!llvm.array<{len(self._message_value)} x i8>")

    @property
    def _message_value(self) -> str:
        return "No suitable guard matched\x00"

    @cached_property
    def _parameters(self) -> Sequence[str]:
        return inspect.signature(self.fn).parameters.keys()

    @property
    def _ptr_type(self) -> ir.Type:
        return llvm.PointerType.get()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        while True:
            try:
                return self.executor(*args, **kwargs)
            except GuardMatchException:
                self.compile(*args, **kwargs)

    def stub_compile(self) -> Callable[..., Any]:
        module: ir.Module = self._build_ir_module()
        return self._build_execution_engine_callable(module)

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        gm, gs = torch._dynamo.export(
            self.fn, aten_graph=True, assume_static_by_default=True
        )(*args, **kwargs)
        ret: Any = gm(*args, **kwargs)
        guards: Guards = parse_guards(gs)
        model_name: str = f"{self.fn.__name__}_{hex(hash(guards))}"
        module: ir.Module = fx.stateless_fx_import(
            gm,
            output_type=compiler_utils.OutputType.TORCH,
            model_name=model_name,
            fx_importer=LibTritonFxImporter(context=self.ctx),
        )
        with self.ctx:
            passmanager.PassManager.parse(
                LibTritonGraphModule._build_torch_pipeline()
            ).run(module.operation)
            wrapper_name, wrapper_type = self._emit_submodule_wrapper(
                module,
                guards,
            )
        self.guard_variants.append(
            GuardVariant(
                guard=guards,
                module=module,
                wrapper_name=wrapper_name,
                wrapper_type=wrapper_type,
            )
        )
        self.executor = self.stub_compile()
        return ret

    @staticmethod
    def _lookup_func_type(
        module: ir.Module, fn_name: str
    ) -> Tuple[str, ir.FunctionType]:
        [op] = [
            op
            for op in module.body.operations
            if isinstance(op, func.FuncOp) and op.sym_name.value.startswith(fn_name)
        ]
        return op.sym_name.value, ir.FunctionType(op.type)

    def _emit_unpack_any_arg(
        self,
        packed_args_ptr: ir.Value,
        index: int,
    ) -> ir.Value:
        arg_slot_ptr: ir.Value = llvm.GEPOp(
            self._ptr_type,
            packed_args_ptr,
            [],
            ir.DenseI32ArrayAttr.get([index]),
            self._any_llvm_type,
            None,
        ).result
        return tvm_ffi_d.load(self._any_type, arg_slot_ptr)

    def _emit_unbox_any_arg(
        self,
        packed_args_ptr: ir.Value,
        index: int,
        target_ty: ir.Type,
    ) -> ir.Value:
        any_value: ir.Value = self._emit_unpack_any_arg(
            packed_args_ptr,
            index,
        )
        if isinstance(target_ty, ir.RankedTensorType):
            dl_tensor: ir.Value = tvm_ffi_d.to(self._dl_tensor_type, any_value)
            memref_ty: ir.Type = ir.MemRefType.get(
                target_ty.shape,
                target_ty.element_type,
            )
            memref: ir.Value = dlpack.to_memref(memref_ty, dl_tensor)
            return bufferization.to_tensor(
                target_ty,
                memref,
                restrict=True,
                writable=True,
            )
        elif isinstance(target_ty, ir.MemRefType):
            dl_tensor = tvm_ffi_d.to(self._dl_tensor_type, any_value)
            return dlpack.to_memref(target_ty, dl_tensor)
        else:
            return tvm_ffi_d.to(target_ty, any_value)

    def _emit_box_return(self, packed_result_ptr: ir.Value, value: ir.Value) -> None:
        tvm_ffi_d.store(value, packed_result_ptr)

    def _emit_dispatch_call(
        self,
        packed_values_ptr: ir.Value,
        packed_args_ptr: ir.Value,
        num_args: ir.Value,
        packed_result_ptr: ir.Value,
        callee_name: str,
        callee_ty: ir.FunctionType,
    ) -> ir.Value:
        assert len(callee_ty.inputs) == 4 and len(callee_ty.results) == 1, (
            f"submodule wrapper signature mismatch: {callee_name}: {callee_ty}"
        )
        call_op: func.CallOp = func.CallOp(
            [self._i32_type],
            callee_name,
            [packed_values_ptr, packed_args_ptr, num_args, packed_result_ptr],
        )
        [status] = call_op.results
        return status

    def _emit_fallback_error(self) -> ir.Value:
        kind_ptr: ir.Value = llvm.AddressOfOp(
            self._ptr_type, self._kind_global_name
        ).result
        message_ptr: ir.Value = llvm.AddressOfOp(
            self._ptr_type, self._message_global_name
        ).result
        tvm_ffi_d.ErrorSetRaisedFromCStrOp(kind_ptr, message_ptr)
        return arith.constant(self._i32_type, -1)

    def _emit_dispatch_if_chain(
        self,
        dispatch_iter: Iterator[Tuple[Guards, str, ir.FunctionType]],
        packed_values_ptr: ir.Value,
        packed_args_ptr: ir.Value,
        num_args: ir.Value,
        packed_result_ptr: ir.Value,
        symbol_table: Optional[Dict[str, ir.Value]] = None,
    ) -> ir.Value:
        item: Optional[Tuple[Guards, str, ir.FunctionType]] = next(dispatch_iter, None)
        if item is None:
            return self._emit_fallback_error()
        if symbol_table is None:
            symbol_table = {
                name: self._emit_unpack_any_arg(packed_args_ptr, idx)
                for idx, name in enumerate(self._parameters)
            }
        guard, callee_name, callee_ty = item
        cond: ir.Value = guard.build_ir(symbol_table, context=self.ctx)
        if_op: scf.IfOp = scf.IfOp(cond, [self._i32_type], has_else=True)
        with ir.InsertionPoint(if_op.then_block):
            success: ir.Value = self._emit_dispatch_call(
                packed_values_ptr,
                packed_args_ptr,
                num_args,
                packed_result_ptr,
                callee_name,
                callee_ty,
            )
            scf.YieldOp([success])
        with ir.InsertionPoint(if_op.else_block):
            fallback: ir.Value = self._emit_dispatch_if_chain(
                dispatch_iter,
                packed_values_ptr,
                packed_args_ptr,
                num_args,
                packed_result_ptr,
                symbol_table,
            )
            scf.YieldOp([fallback])
        [result] = if_op.results
        return result

    def _emit_submodule_wrapper(
        self,
        module: ir.Module,
        guards: Guards,
    ) -> Tuple[str, ir.FunctionType]:
        callee_name: str
        callee_ty: ir.FunctionType
        callee_name, callee_ty = LibTritonGraphModule._lookup_func_type(
            module,
            self.fn.__name__,
        )
        num_user_args: int = len(self._parameters)
        assert len(callee_ty.inputs) >= num_user_args, (
            f"submodule input arity mismatch: {callee_name}: {callee_ty}"
        )
        call_input_tys: Sequence[ir.Type] = callee_ty.inputs[:num_user_args]
        out_param_tys: Sequence[ir.Type] = callee_ty.inputs[num_user_args:]
        if len(out_param_tys) == 0:
            assert len(callee_ty.results) == 1, (
                "submodule must either expose exactly one return or one out-param: "
                f"{callee_name}: {callee_ty}"
            )
        else:
            assert len(out_param_tys) == 1 and len(callee_ty.results) == 0, (
                "submodule out-param arity > 1 is unsupported: "
                f"{callee_name}: {callee_ty}"
            )
            [out_param_ty] = out_param_tys
            assert isinstance(out_param_ty, ir.MemRefType), (
                f"submodule out-param type must be memref: {callee_name}: {callee_ty}"
            )
        wrapper_name: str = f"__tvm_ffi_{self.fn.__name__}_guard_{hex(hash(guards))}"
        with ir.InsertionPoint(module.body), ir.Location.unknown():
            wrapper_op: func.FuncOp = func.FuncOp(wrapper_name, self._func_type)
            entry_block: ir.Block = ir.Block.create_at_start(
                wrapper_op.body,
                [self._ptr_type, self._ptr_type, self._i32_type, self._ptr_type],
            )
            _, packed_args_ptr, _, packed_result_ptr = entry_block.arguments
            with ir.InsertionPoint(entry_block):
                call_args: List[ir.Value] = [
                    self._emit_unbox_any_arg(packed_args_ptr, idx, target_ty)
                    for idx, target_ty in enumerate(call_input_tys)
                ]
                if len(out_param_tys) == 0:
                    call_op: func.CallOp = func.CallOp(
                        callee_ty.results,
                        callee_name,
                        call_args,
                    )
                    [result] = call_op.results
                    self._emit_box_return(packed_result_ptr, result)
                    func.return_([arith.constant(self._i32_type, 0)])
                else:
                    [out_param_ty] = out_param_tys
                    out_handle: ir.Value = tvm_ffi_d.env_tensor_alloc(
                        self._object_handle_type,
                        out_param_ty.element_type,
                        out_param_ty.shape,
                    )
                    opaque_ptr: ir.Value = tvm_ffi_d.get_opaque_ptr(
                        self._ptr_type,
                        out_handle,
                    )
                    dl_tensor: ir.Value = tvm_ffi_d.load(
                        self._dl_tensor_type,
                        opaque_ptr,
                    )
                    out_memref: ir.Value = dlpack.to_memref(
                        out_param_ty,
                        dl_tensor,
                    )
                    call_op: func.CallOp = func.CallOp(
                        callee_ty.results,
                        callee_name,
                        call_args + [out_memref],
                    )
                    assert len(call_op.results) == 0
                    self._emit_box_return(packed_result_ptr, out_handle)
                    func.return_([arith.constant(self._i32_type, 0)])
        return wrapper_name, ir.FunctionType(wrapper_op.type)

    def _build_ir_module(self) -> ir.Module:
        """Build an IR module with a guard-dispatch TVM-FFI wrapper.

        The wrapper iterates all compiled guard variants and emits a conditional
        dispatch chain. On the first matched guard it forwards packed call
        buffers to the corresponding submodule TVM-FFI wrapper and returns the
        submodule status code. If no guard matches, it sets the
        GuardMatchException and returns -1.
        """
        with self.ctx, ir.Location.unknown():
            module: ir.Module = ir.Module.create()
            module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
            for variant in self.guard_variants:
                transform.interpreter.copy_symbols_and_merge_into(
                    module, variant.module
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
                func_op: func.FuncOp = func.FuncOp(self._func_name, self._func_type)
                entry_block: ir.Block = ir.Block.create_at_start(
                    func_op.body,
                    [self._ptr_type, self._ptr_type, self._i32_type, self._ptr_type],
                )
                (
                    packed_values_ptr,
                    packed_args_ptr,
                    num_args,
                    packed_result_ptr,
                ) = entry_block.arguments
                with ir.InsertionPoint(entry_block):
                    status: ir.Value = self._emit_dispatch_if_chain(
                        map(
                            lambda variant: (
                                variant.guard,
                                variant.wrapper_name,
                                variant.wrapper_type,
                            ),
                            sorted(
                                self.guard_variants,
                                key=lambda variant: hash(variant.guard),
                            ),
                        ),
                        packed_values_ptr,
                        packed_args_ptr,
                        num_args,
                        packed_result_ptr,
                    )
                    func.return_([status])
            major, minor = torch.cuda.get_device_capability()
            pipeline: str = LibTritonGraphModule._build_builtin_pipeline(
                chip=f"sm_{major}{minor}"
            )
            # print(module)
            passmanager.PassManager.parse(pipeline).run(module.operation)
            return module

    def _build_execution_engine_callable(
        self,
        module: ir.Module,
    ) -> Callable[..., Any]:
        engine: execution_engine.ExecutionEngine = execution_engine.ExecutionEngine(
            module,
            shared_libs=capi_utils.find_runtime_libraries(),
        )

        engine.initialize()
        # Find the unique dispatch function symbol.
        [symbol] = [
            op.sym_name.value
            for op in module.body.operations
            if isinstance(op, llvm.LLVMFuncOp) and op.sym_name.value == self._func_name
        ]
        ptr: Callable[..., Any] = engine.raw_lookup(symbol)
        assert ptr is not None, f"symbol not found: {symbol}"
        fn: Callable[..., Any] = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr, keep_alive_object=engine
        )

        def f(*args: Any) -> Any:
            result = fn(*args)
            if isinstance(result, (list, tuple)):
                return tuple(
                    LibTritonGraphModule._canonicalize_ret(value) for value in result
                )
            else:
                return LibTritonGraphModule._canonicalize_ret(result)

        return f

    @staticmethod
    def _build_builtin_pipeline(chip: str) -> str:
        passes: List[str] = [
            "func.func(canonicalize,sccp,canonicalize,cse)",
            "func.func(gpu-map-parallel-loops)",
            "convert-parallel-loops-to-gpu",
            "gpu-kernel-outlining",
            "finalize-memref-to-llvm{use-generic-functions}",
            f"nvvm-attach-target{{O=3 chip={chip}}}",
            "gpu.module(convert-gpu-to-nvvm{index-bitwidth=64})",
            "gpu-module-to-binary{format=fatbin}",
            "convert-scf-to-cf",
            "convert-arith-to-llvm",
            "convert-cf-to-llvm",
            "convert-func-to-llvm",
            "gpu-to-llvm",
            "convert-to-llvm",
            "func.func(canonicalize, cse)",
            "reconcile-unrealized-casts",
        ]
        return "builtin.module({})".format(", ".join(passes))

    @staticmethod
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
            "func.func(canonicalize,cse)",
            "func.func(torch-finalizing-backend-type-conversion)",
            "func.func(canonicalize,sccp,canonicalize,cse)",
            "one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map}",
            "buffer-results-to-out-params{hoist-dynamic-allocs hoist-static-allocs modify-public-functions}",
            "convert-linalg-to-parallel-loops",
        ]
        return "builtin.module({})".format(", ".join(passes))

    @staticmethod
    def _canonicalize_ret(val: Any) -> Any:
        return (
            torch.from_dlpack(val)
            if hasattr(val, "__dlpack__") and not isinstance(val, torch.Tensor)
            else val
        )
