from __future__ import annotations

from typing import Any, Callable, Final, List, Tuple, Type, Union

import torch
import tvm_ffi

from libtriton._C.libtriton_core import (
    compiler_utils,
    fx,
    ir,
    passmanager,
)
from libtriton._C.libtriton_core.dialects import (
    func,
    llvm,
)
from .guards import Guards, parse_guards
from .importer import LibTritonFxImporter
from .signature import Signature


class Function(object):
    """A compiled function variant associated with guard conditions."""

    def __init__(
        self,
        guards: Guards,
        module: ir.Module,
        func_op: func.FuncOp,
        signature: Signature,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.guards: Final[Guards] = guards
        self.module: Final[ir.Module] = module
        self.func_op: Final[func.FuncOp] = func_op
        self.signature: Final[Signature] = signature

    @classmethod
    def build(
        cls: Type[Function],
        fn: Callable[..., Any],
        ctx: ir.Context,
    ) -> Callable[..., Tuple[Any, Function]]:
        """Build a Function from a Python function.

        Returns a callable that, when invoked with user arguments, performs
        torch.export, FX-to-MLIR import, torch lowering pipeline, submodule
        wrapper emission, and returns ``(result, Function)``.
        """

        def f(*args: Any) -> Tuple[Any, Function]:
            fn_name: Final[str] = fn.__name__
            gm, gs = torch._dynamo.export(
                fn, aten_graph=True, assume_static_by_default=True
            )(*args)
            ret: Any = gm(*args)
            inputs_schema: List[tvm_ffi.core.TypeSchema] = [
                tvm_ffi.core.TypeSchema.from_annotation(type(arg)) for arg in args
            ]
            if isinstance(ret, (tuple, list)):
                outputs_schema: Union[
                    tvm_ffi.core.TypeSchema, List[tvm_ffi.core.TypeSchema]
                ] = [tvm_ffi.core.TypeSchema.from_annotation(type(r)) for r in ret]
            else:
                outputs_schema = tvm_ffi.core.TypeSchema.from_annotation(type(ret))
            guards: Guards = parse_guards(gs)
            model_name: str = f"{fn_name}_{hex(hash(guards))}"
            module: ir.Module = fx.stateless_fx_import(
                gm,
                output_type=compiler_utils.OutputType.TORCH,
                model_name=model_name,
                fx_importer=LibTritonFxImporter(context=ctx),
            )
            with ctx, ir.Location.unknown():
                passmanager.PassManager.parse(Function._build_torch_pipeline()).run(
                    module.operation
                )
                sig: Signature = Signature(
                    outputs_schema=outputs_schema,
                    inputs_schema=inputs_schema,
                )
                func_op: func.FuncOp = Function._emit_submodule_wrapper(
                    module,
                    guards,
                    fn_name,
                    sig,
                )
            return ret, cls(
                guards=guards,
                module=module,
                func_op=func_op,
                signature=sig,
            )

        return f

    @staticmethod
    def _build_torch_pipeline() -> str:
        """Build the torch-to-MLIR lowering pipeline string."""
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
            "buffer-results-to-out-params{add-result-attr hoist-dynamic-allocs hoist-static-allocs modify-public-functions}",
            "convert-linalg-to-parallel-loops",
        ]
        return "builtin.module({})".format(", ".join(passes))

    @staticmethod
    def _lookup_func_type(module: ir.Module, fn_name: str) -> func.FuncOp:
        [op] = [
            op
            for op in module.body.operations
            if isinstance(op, func.FuncOp) and op.sym_name.value.startswith(fn_name)
        ]
        return op

    @staticmethod
    def _emit_submodule_wrapper(
        module: ir.Module,
        guards: Guards,
        fn_name: str,
        sig: Signature,
    ) -> func.FuncOp:
        """Emit a TVM-FFI wrapper that bridges packed args → submodule call.

        Creates the FFI-callable function, delegates the core call
        logic (unpack, allocate out-params, call, box, store) to *sig*,
        and returns the wrapper ``func.FuncOp``.
        """
        i32_type: ir.Type = ir.IntegerType.get_signless(32)
        ptr_type: ir.Type = llvm.PointerType.get()

        callee_op: func.FuncOp = Function._lookup_func_type(module, fn_name)

        # -- Emit wrapper MLIR ------------------------------------------
        wrapper_name: str = f"__tvm_ffi_{fn_name}_{hex(hash(guards))}"
        func_type: ir.FunctionType = ir.FunctionType.get(
            [ptr_type, ptr_type, i32_type, ptr_type], [i32_type]
        )
        with ir.InsertionPoint(module.body), ir.Location.unknown():
            func_op: func.FuncOp = func.FuncOp(wrapper_name, func_type)
            entry_block: ir.Block = ir.Block.create_at_start(
                func_op.body,
                [ptr_type, ptr_type, i32_type, ptr_type],
            )
            _, packed_args_ptr, _, packed_result_ptr = entry_block.arguments
            with ir.InsertionPoint(entry_block):
                status: ir.Value = sig.call(
                    callee_op,
                    packed_args_ptr,
                    packed_result_ptr,
                )
                func.return_([status])
        return func_op
