# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import gc
import inspect
import operator
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Final, TypeAlias

import torch
import tvm_ffi
import tvm_ffi.utils
from torch.export._trace import _extract_fake_inputs
from torch.export.graph_signature import (
    InputKind,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
)
from torch.utils import _pytree as pytree

from trident import capi_utils
from trident.core import (
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from trident.core.dialects import (
    func,
    llvm,
    transform,
)
from trident.core.dialects import (
    tvm_ffi as tvm_ffi_d,
)
from trident.core.execution_engine import ExecutionEngine
from trident.ffi import Exception

from .guards import parse_guards
from .importer import TridentFxImporter
from .input import InputTableBuilder
from .ir import array, box

RuntimeValue: TypeAlias = Any
RuntimeArguments: TypeAlias = tuple[RuntimeValue, ...]
RuntimeKeywords: TypeAlias = dict[str, RuntimeValue]
RuntimeCallable: TypeAlias = Callable[..., RuntimeValue]


class TridentGraphModule:
    """Imports a Torch FX graph into MLIR and executes through TVM FFI.

    Maintains a list of sub-modules, one per unique guard specialization
    encountered at call time.  On each ``compile()`` call the class builds
    a fresh combined module that contains every sub-module plus an LLVM-level
    dispatcher.  The dispatcher tries each sub-module's compiled function in
    order - the first one that returns 0 (guard match) wins.
    """

    def __call__(self, *args: RuntimeValue, **kwargs: RuntimeValue) -> RuntimeValue:
        result = self.executor(*args, **kwargs)
        if isinstance(result, Exception):
            # The dispatcher returned an Exception ObjectRef
            # (all specializations failed); compile a new one.
            return self.compile(*args, **kwargs)
        else:
            return result

    def __init__(
        self,
        fn: RuntimeCallable,
        *,
        dynamic: bool = True,
    ) -> None:
        super().__init__()
        self.fn: Final[RuntimeCallable] = fn
        self.dynamic: Final[bool] = dynamic
        self.ctx: Final[ir.Context] = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()
        self._sub_modules: list[ir.Module] = []
        self.executor: RuntimeCallable = self.stub_compile()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __name__(self) -> str:
        return self.fn.__name__

    def _alloca_tvmffi_any(
        self,
        index: int,
        payload: ir.Value,
    ) -> ir.Value:
        """Allocate a TVMFFIAny slot, fill it, and return the pointer.

        Initializes the slot with a complete ABI struct.
        """
        i32_ty: ir.Type = ir.IntegerType.get_signless(32)
        i64_ty: ir.Type = ir.IntegerType.get_signless(64)
        slot: ir.Value = llvm.alloca(
            res=llvm.PointerType.get(),
            array_size=llvm.mlir_constant(
                value=ir.IntegerAttr.get(i64_ty, 1),
            ),
            elem_type=ir.TypeAttr.get(
                llvm.StructType.get_literal([i32_ty, i32_ty, i64_ty], context=self.ctx),
            ),
        )
        llvm.store(self._build_tvmffi_any(index, payload), slot)
        return slot

    def _build_combined_module(self) -> ir.Module:
        """Merge all sub-modules into a single ``ir.Module``.

        Uses ``copy_symbols_and_merge_into`` which clones each sub-module
        internally, avoiding hangs caused by repeated merging of the same
        module objects.
        """
        # Derive module location from the function being compiled so
        # that the combined module has a meaningful source anchor.
        fn_file: str = inspect.getfile(self.fn)
        _, fn_line = inspect.getsourcelines(self.fn)
        combined: ir.Module = ir.Module.create(
            loc=ir.Location.file(fn_file, fn_line, col=0, context=self.ctx)
        )
        for sub_mod in self._sub_modules:
            with self.ctx, sub_mod.operation.location:
                for attr_name in sub_mod.operation.attributes:
                    if (
                        attr_name not in combined.operation.attributes
                        and not attr_name.startswith("sym_name")
                    ):
                        combined.operation.attributes[attr_name] = (
                            sub_mod.operation.attributes[attr_name]
                        )

                transform.interpreter.copy_symbols_and_merge_into(
                    combined.operation,
                    sub_mod.operation,
                )
        return combined

    # ------------------------------------------------------------------ #
    # Internal: LLVM dispatcher
    # ------------------------------------------------------------------ #

    def _build_llvm_dispatcher(self, module: ir.Module) -> None:
        """Try packed specializations in order, releasing guard failures.

        An Exception result requests the next specialization; a nonzero packed
        call status propagates the runtime error without inspecting its result.
        """
        with self.ctx, module.operation.location, ir.InsertionPoint(module.body):
            i32 = ir.IntegerType.get_signless(32)
            i64 = ir.IntegerType.get_signless(64)
            pointer = llvm.PointerType.get()
            any_type = llvm.StructType.get_literal([i32, i32, i64])
            name_type = llvm.StructType.get_literal([pointer, i64])
            packed_type = ir.Type.parse(
                "!llvm.func<i32 (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr)>"
            )
            symbols = ir.SymbolTable(module.operation)
            declarations = {
                "TVMFFIFunctionCall": packed_type,
                "TVMFFIFunctionGetGlobal": ir.Type.parse(
                    "!llvm.func<i32 (!llvm.ptr, !llvm.ptr)>"
                ),
                "TVMFFIObjectDecRef": ir.Type.parse("!llvm.func<i32 (!llvm.ptr)>"),
            }
            for name, type_ in declarations.items():
                if name not in symbols:
                    llvm.func(name, ir.TypeAttr.get(type_))
            dispatcher = llvm.func(
                f"__tvm_ffi_{self.fn.__name__}", ir.TypeAttr.get(packed_type)
            )
            entry = ir.Block.create_at_start(
                dispatcher.body, [pointer, pointer, i32, pointer]
            )
            in_ptr, out_ptr, num_args, ret_ptr = entry.arguments
            error = ir.Block.create_after(entry)
            done = ir.Block.create_after(error)
            exit_ = ir.Block.create_after(done, i32)

            def call(name: str, arguments: Sequence[ir.Value]) -> ir.Value:
                return llvm.call(i32, arguments, [], [], callee=name)

            def field(slot: ir.Value, index: int, type_: ir.Type) -> ir.Value:
                return llvm.extractvalue(type_, llvm.load(any_type, slot), [index])

            def string(name: str, value: str) -> ir.Value:
                with ir.InsertionPoint(module.body):
                    llvm.mlir_global(
                        ir.TypeAttr.get(
                            ir.Type.parse(f"!llvm.array<{len(value)} x i8>")
                        ),
                        name,
                        ir.Attribute.parse("#llvm.linkage<internal>"),
                        constant=True,
                        value=ir.StringAttr.get(value),
                    )
                return llvm.mlir_addressof(pointer, name)

            with ir.InsertionPoint(entry):
                zero = llvm.mlir_constant(ir.IntegerAttr.get(i32, 0))
                one = llvm.mlir_constant(ir.IntegerAttr.get(i32, 1))
                zero_i64 = llvm.mlir_constant(ir.IntegerAttr.get(i64, 0))
                one_i64 = llvm.mlir_constant(ir.IntegerAttr.get(i64, 1))
                null = llvm.mlir_zero(pointer)
                handles = {}
                for name in ("trident.ffi.Exception", "trident.ffi.GetExceptionIndex"):
                    name_ptr = string(f"__trident_const_{name}", name)
                    name_value = llvm.mlir_undef(name_type)
                    name_value = llvm.insertvalue(name_value, name_ptr, [0])
                    name_value = llvm.insertvalue(
                        name_value,
                        llvm.mlir_constant(ir.IntegerAttr.get(i64, len(name))),
                        [1],
                    )
                    name_slot = llvm.alloca(pointer, one_i64, name_type)
                    llvm.store(name_value, name_slot)
                    handle_slot = llvm.alloca(pointer, one_i64, pointer)
                    call("TVMFFIFunctionGetGlobal", [name_slot, handle_slot])
                    handles[name] = llvm.load(pointer, handle_slot)
                index_slot = self._alloca_tvmffi_any(0, zero_i64)
                call(
                    "TVMFFIFunctionCall",
                    [handles["trident.ffi.GetExceptionIndex"], null, zero, index_slot],
                )
                exception_index = llvm.trunc(
                    i32,
                    field(index_slot, 2, i64),
                    ir.Attribute.parse("#llvm.overflow<none>"),
                )
                result_slot = self._alloca_tvmffi_any(0, zero_i64)
                kind = string(
                    "__trident_const_GuardMatchException", "GuardMatchException\0"
                )
                arguments = self._alloca_tvmffi_any(8, llvm.ptrtoint(i64, kind))
                call(
                    "TVMFFIFunctionCall",
                    [handles["trident.ffi.Exception"], arguments, one, result_slot],
                )

            previous = entry
            for index in range(len(self._sub_modules)):
                attempt = ir.Block.create_after(previous)
                check = ir.Block.create_after(attempt)
                with ir.InsertionPoint(previous):
                    failed = llvm.icmp(
                        llvm.ICmpPredicate.eq,
                        field(result_slot, 0, i32),
                        exception_index,
                    )
                    llvm.cond_br(failed, [], [], attempt, done)
                with ir.InsertionPoint(attempt):
                    call(
                        "TVMFFIObjectDecRef",
                        [llvm.inttoptr(pointer, field(result_slot, 2, i64))],
                    )
                    status = call(
                        f"__tvm_ffi_{self.fn.__name__}_{index}",
                        [in_ptr, out_ptr, num_args, result_slot],
                    )
                    success = llvm.icmp(llvm.ICmpPredicate.eq, status, zero)
                    llvm.cond_br(success, [], [], check, error)
                previous = check
            with ir.InsertionPoint(previous):
                llvm.br([], done)
            with ir.InsertionPoint(error):
                llvm.br([llvm.mlir_constant(ir.IntegerAttr.get(i32, -1))], exit_)
            with ir.InsertionPoint(done):
                llvm.store(llvm.load(any_type, result_slot), ret_ptr)
                llvm.br([zero], exit_)
            with ir.InsertionPoint(exit_):
                for handle in handles.values():
                    call("TVMFFIObjectDecRef", [handle])
                llvm.return_(arg=exit_.arguments[0])

    # ------------------------------------------------------------------ #
    # Internal: sub-module construction
    # ------------------------------------------------------------------ #

    def _build_sub_module(
        self,
        args: list[RuntimeValue],
        kwargs: RuntimeKeywords,
    ) -> RuntimeValue:
        """Export -> import -> wrap a single sub-module for *args*.

        Each sub-module's ``func.func`` is named ``main_{index}`` and its
        ``tvm_ffi.func`` is named ``{self.fn.__name__}_{index}`` to avoid symbol
        collisions in the merged module.
        """

        # Step 1: Export  ---------------------------------------------------
        index = len(self._sub_modules)

        signature: inspect.Signature = inspect.signature(self.fn)
        trace_args: RuntimeArguments = tuple(args)
        bound: inspect.BoundArguments = signature.bind(*trace_args, **kwargs)
        bound.apply_defaults()
        exported_gm, gs = torch._dynamo.export(
            self.fn,
            aten_graph=True,
            assume_static_by_default=not self.dynamic,
        )(*trace_args, **kwargs)
        dynamic_shapes = self._retrieve_dynamic_shapes(exported_gm, trace_args, kwargs)
        trace_gm = copy.deepcopy(exported_gm)

        exported_program: torch.export.ExportedProgram = torch.export.export(
            trace_gm,
            trace_args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        warmup_result = torch.fx.Interpreter(exported_program.module()).run(
            *pytree.tree_leaves(exported_program.example_inputs)
        )

        self._output_spec = pytree.tree_structure(warmup_result)

        # Step 2: Import FX -> MLIR  ----------------------------------------
        importer: TridentFxImporter = TridentFxImporter(
            context=self.ctx, specialization_id=index
        )
        main_func_name: Final[str] = f"main_{index}"
        main_func: func.FuncOp = importer.import_program(
            exported_program, func_name=main_func_name
        )
        with self.ctx:
            main_func.attributes["sym_visibility"] = ir.StringAttr.get("private")
        module: ir.Module = importer.module

        torch._dynamo.reset()

        # Step 3: Wrap with tvm_ffi.func.
        flat_types: Sequence[ir.Type] = main_func.type.inputs

        with self.ctx:
            input_builder = InputTableBuilder.get(
                exported_program,
                signature,
                bound.arguments,
                flat_types,
            )
        param_types = input_builder.input_types

        with self.ctx:
            abi_result_type = tvm_ffi_d.AnyType.get(context=self.ctx)
            ffi_type: ir.FunctionType = ir.FunctionType.get(
                param_types, [abi_result_type]
            )

        tvm_ffi_name: Final[str] = f"{self.fn.__name__}_{index}"
        with ir.InsertionPoint(module.body), main_func.operation.location:
            ffi_func: tvm_ffi_d.FuncOp = tvm_ffi_d.func(
                tvm_ffi_name,
                ir.TypeAttr.get(ffi_type),
                emit_tvm_ffi_abi=ir.UnitAttr.get(self.ctx),
            )
            entry_block: ir.Block = ir.Block.create_at_start(ffi_func.body, param_types)
            with ir.InsertionPoint(entry_block):
                success_block, failure_block = parse_guards(gs).build(
                    lambda: input_builder.build(entry_block.arguments),
                    self.ctx,
                    entry_block,
                )
                # main_* is an ordinary func.func. Only the surrounding
                # tvm_ffi.func is exposed through the TVM FFI ABI. Guards are
                # already represented as CFG blocks, so ownership analysis can
                # see their short-circuit edges directly.
                with ir.InsertionPoint(success_block):
                    main_table = input_builder.build(entry_block.arguments)
                    input_specs = exported_program.graph_signature.input_specs
                    flat_inputs = main_table.flatten_inputs()
                    assert len(flat_inputs) == len(input_specs), (
                        "unexpected number of reconstructed exported inputs: "
                        f"got {len(flat_inputs)}, expected {len(input_specs)}"
                    )
                    main_inputs = [
                        value
                        for input_spec, value in zip(input_specs, flat_inputs)
                        if input_spec.kind == InputKind.USER_INPUT
                        and isinstance(
                            input_spec.arg,
                            (
                                TensorArgument,
                                SymBoolArgument,
                                SymFloatArgument,
                                SymIntArgument,
                            ),
                        )
                    ]
                    assert len(main_inputs) == len(flat_types), (
                        "unexpected number of reconstructed MLIR inputs: "
                        f"got {len(main_inputs)}, expected {len(flat_types)}"
                    )
                    main_args = main_inputs
                    call_result: ir.Value | Sequence[ir.Value] = func.call(
                        main_func.type.results,
                        main_func_name,
                        main_args,
                    )
                    result = (
                        call_result
                        if isinstance(call_result, ir.Value)
                        else array(list(call_result))
                    )
                    tvm_ffi_d.return_([tvm_ffi_d.cast(abi_result_type, box(result))])

                with ir.InsertionPoint(failure_block):
                    exception = tvm_ffi_d.exception("GuardMatch")
                    error_value = tvm_ffi_d.cast(abi_result_type, exception)
                    tvm_ffi_d.return_([error_value])

        self._sub_modules.append(module)
        self.executor = self.stub_compile()
        gc.collect()
        return warmup_result

    # ------------------------------------------------------------------ #
    # Internal: TVMFFIAny helpers
    # ------------------------------------------------------------------ #

    def _build_tvmffi_any(self, index: int, payload: ir.Value) -> ir.Value:
        """Build the ABI struct containing a type index, padding and payload."""
        i32 = ir.IntegerType.get_signless(32)
        i64 = ir.IntegerType.get_signless(64)
        result = llvm.mlir_undef(llvm.StructType.get_literal([i32, i32, i64]))
        fields = (
            llvm.mlir_constant(ir.IntegerAttr.get(i32, index)),
            llvm.mlir_constant(ir.IntegerAttr.get(i32, 0)),
            payload,
        )
        for position, value in enumerate(fields):
            result = llvm.insertvalue(result, value, [position])
        return result

    # ------------------------------------------------------------------ #
    # Internal: dynamic-shape helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _retrieve_dynamic_shapes(
        exported_gm: torch.fx.GraphModule,
        args: Sequence[RuntimeValue],
        kwargs: RuntimeKeywords,
    ) -> torch.export.ShapesCollection:
        """Recover a ``torch.export`` shape specification from Dynamo metadata."""
        fake_args, fake_kwargs, fake_mode = _extract_fake_inputs(
            exported_gm, tuple(args), kwargs
        )
        shape_env = fake_mode.shape_env
        assert shape_env is not None, "Dynamo FakeTensorMode does not have a ShapeEnv"

        dimensions: dict[Hashable, torch.export.Dim] = {}
        dynamic_shapes = torch.export.ShapesCollection()

        def dimension(symbol: Hashable) -> torch.export.Dim:
            result = dimensions.get(symbol)
            if result is not None:
                return result

            value_range = shape_env.var_to_range.get(symbol)
            options: dict[str, int] = {}
            if value_range is not None:
                if value_range.lower.is_Integer is True:
                    options["min"] = operator.index(round(value_range.lower))
                if value_range.upper.is_Integer is True:
                    options["max"] = operator.index(round(value_range.upper))
            result = torch.export.Dim.AUTO(**options)
            dimensions[symbol] = result
            return result

        def register_shape(value: Any, fake_value: Any) -> None:
            if isinstance(value, torch.Tensor):
                assert isinstance(fake_value, torch.Tensor), (
                    "Dynamo placeholder for a Tensor input does not contain "
                    "FakeTensor metadata"
                )
                assert value.dim() == fake_value.dim(), (
                    f"Dynamo FakeTensor has rank {fake_value.dim()}, but its input "
                    f"has rank {value.dim()}"
                )
                dynamic_shapes[value] = {
                    index: dimension_spec
                    for index, size in enumerate(fake_value.shape)
                    if isinstance(size, torch.SymInt)
                    if (dimension_spec := symbolic_dimension(size, value.shape[index]))
                    is not None
                }

        def symbolic_dimension(
            value: torch.SymInt, concrete_size: int
        ) -> torch.export.Dim | None:
            expression = value.node.expr
            symbols = expression.free_symbols
            if not symbols:
                assert expression.is_Integer is True, (
                    f"Dynamo dimension {expression} is neither static nor symbolic"
                )
                assert operator.index(round(expression)) == concrete_size, (
                    f"Dynamo dimension {expression} does not match input size "
                    f"{concrete_size}"
                )
                return None
            assert len(symbols) == 1, (
                f"cannot represent Dynamo dimension {expression} with torch.export.Dim"
            )

            [symbol] = symbols
            coefficient = expression.coeff(symbol)
            offset = expression.subs(symbol, 0)
            assert (
                coefficient.is_Integer is True
                and coefficient > 0
                and offset.is_Integer is True
                and expression == coefficient * symbol + offset
            ), f"cannot represent Dynamo dimension {expression} with torch.export.Dim"

            result = dimension(symbol)
            if coefficient != 1:
                result = operator.index(round(coefficient)) * result
            if offset != 0:
                result = result + operator.index(round(offset))
            return result

        pytree.tree_map(
            register_shape,
            (tuple(args), kwargs),
            (fake_args, fake_kwargs),
        )
        return dynamic_shapes

    def compile(self, *args: RuntimeValue, **kwargs: RuntimeValue) -> RuntimeValue:
        """Build a new sub-module for *args* and rebuild the combined
        module + dispatcher.  Called automatically from ``__call__`` when a
        ``GuardMatchException`` is raised."""
        return self._build_sub_module([*args], {**kwargs})

    # ------------------------------------------------------------------ #
    # Internal: orchestration
    # ------------------------------------------------------------------ #

    def stub_compile(self) -> RuntimeCallable:
        """Build the combined module, lower it, add the LLVM dispatcher,
        JIT-compile, and return a callable executor.

        Called initially from ``__init__`` (with zero sub-modules) and
        after every ``compile()`` to pick up newly added specializations.
        """
        # 1. Merge all sub-modules into one (each is cloned to avoid
        #    repeated-merging hangs).
        combined: ir.Module = self._build_combined_module()

        # 2. Lower semantic TVM FFI and GPU IR to LLVM.
        with self.ctx:
            passmanager.PassManager.parse(
                "builtin.module(trident-lowering-pipeline)",
            ).run(combined.operation)

        # 3. Build the LLVM dispatcher that tries each sub-function.
        self._build_llvm_dispatcher(combined)

        # 4. JIT-compile and wrap.
        engine: ExecutionEngine = ExecutionEngine(
            combined,
            opt_level=3,
            shared_libs=capi_utils.find_runtime_libraries(),
        )
        engine.initialize()

        symbol: str = f"__tvm_ffi_{self.fn.__name__}"
        ptr: int = engine.raw_lookup(symbol)
        assert ptr is not None, (
            f"symbol not found: {symbol}; "
            f"available: {[op.sym_name.value for op in combined.body.operations if hasattr(op, 'sym_name')]}"
        )

        fn: tvm_ffi.Function = tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr,
            keep_alive_object=engine,
        )

        def invoke(*values: RuntimeValue) -> RuntimeValue:
            result = normalize(
                fn(
                    *(
                        normalize(
                            value,
                            lambda element: (
                                tvm_ffi.device(f"{element}")
                                if isinstance(element, torch.device)
                                else tvm_ffi.convert(element)
                                if isinstance(element, torch.dtype)
                                else element
                            ),
                        )
                        for value in values
                    )
                ),
                lambda value: (
                    torch.from_dlpack(value)
                    if isinstance(value, tvm_ffi.Tensor)
                    else value
                ),
            )

            if isinstance(result, Exception):
                return result
            leaves = [result] if self._output_spec.num_leaves == 1 else list(result)
            return pytree.tree_unflatten(leaves, self._output_spec)

        def normalize(
            value: RuntimeValue,
            convert: Callable[[RuntimeValue], RuntimeValue],
        ) -> RuntimeValue:
            if isinstance(value, tvm_ffi.Array):
                return [normalize(element, convert) for element in value]
            if isinstance(value, tuple):
                return tuple(normalize(element, convert) for element in value)
            if isinstance(value, list):
                return [normalize(element, convert) for element in value]
            if isinstance(value, dict):
                return {
                    key: normalize(element, convert) for key, element in value.items()
                }
            return convert(value)

        signature = inspect.signature(self.fn)
        ffi_signature = signature.replace(
            parameters=[
                parameter.replace(
                    default=(
                        tvm_ffi.device(f"{parameter.default}")
                        if isinstance(parameter.default, torch.device)
                        else tvm_ffi.convert(parameter.default)
                    )
                )
                if isinstance(parameter.default, (torch.device, torch.dtype))
                else parameter
                for parameter in signature.parameters.values()
            ]
        )
        return tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper_from_signature(
            invoke, ffi_signature
        )

    # ------------------------------------------------------------------ #
    # Internal: module merging
    # ------------------------------------------------------------------ #
