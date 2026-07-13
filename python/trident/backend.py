# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations
from typing import Any, Callable, Dict, Final, List, Tuple

import inspect
import torch
import tvm_ffi
import tvm_ffi.utils

from trident._C.trident_core import (
    capi_utils,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from trident._C.trident_core.dialects import (
    func,
    llvm,
    transform,
    tvm_ffi as tvm_ffi_d,
)
from trident._C.trident_core.execution_engine import ExecutionEngine
from trident._C.trident_core.extras.fx_importer import FxImporter
from .guards import parse_guards
from .patch import apply_patch


class TridentGraphModule(object):
    """Compiles a torch function via Torch-MLIR and wraps with tvm_ffi.

    Maintains a list of sub-modules, one per unique guard specialization
    encountered at call time.  On each ``compile()`` call the class builds
    a fresh combined module that contains every sub-module plus an LLVM-level
    dispatcher.  The dispatcher tries each sub-module's compiled function in
    order - the first one that returns 0 (guard match) wins.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        max_compiles: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Callable[..., Any]] = fn
        self._max_compiles: Final[int] = max_compiles
        self.ctx: Final[ir.Context] = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()
        self._sub_modules: List[ir.Module] = []
        self.executor: Callable[..., Any] = self.stub_compile()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(self, *args: Any, **kwargs: Any) -> Any:

        for _ in range(self._max_compiles):
            result = self.executor(*args, **kwargs)
            if result.__class__.__name__ == "Exception":
                # The dispatcher returned an Exception ObjectRef
                # (all specializations failed); compile a new one.
                self.compile(*args, **kwargs)
            else:
                return result

        raise RuntimeError(
            f"recompilation limit ({self._max_compiles}) exceeded without finding a matching specialization"
        )

    def compile(self, *args: Any, **kwargs) -> Any:
        """Build a new sub-module for *args* and rebuild the combined
        module + dispatcher.  Called automatically from ``__call__`` when a
        ``GuardMatchException`` is raised."""
        # 1. Build a new sub-module for the current arguments.
        sub_mod: ir.Module = self._build_sub_module(
            self.fn, self.ctx, len(self._sub_modules), args, kwargs
        )
        self._sub_modules.append(sub_mod)

        # 2. Rebuild the combined module and dispatcher.
        self.executor = self.stub_compile()

    # ------------------------------------------------------------------ #
    # Internal: orchestration
    # ------------------------------------------------------------------ #

    def stub_compile(self) -> Callable[..., Any]:
        """Build the combined module, lower it, add the LLVM dispatcher,
        JIT-compile, and return a callable executor.

        Called initially from ``__init__`` (with zero sub-modules) and
        after every ``compile()`` to pick up newly added specializations.
        """
        # 1. Merge all sub-modules into one (each is cloned to avoid
        #    repeated-merging hangs).
        combined: ir.Module = self._build_combined_module()

        # 2. Lower Torch / TVM-FFI -> LLVM.
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

        f: Callable = tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper_from_signature(
            fn, inspect.signature(self.fn)
        )

        return f

    # ------------------------------------------------------------------ #
    # Internal: sub-module construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_sub_module(
        fn: Callable[..., Any],
        ctx: ir.Context,
        index: int,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> ir.Module:
        """Export -> import -> wrap a single sub-module for *args*.

        Each sub-module's ``func.func`` is named ``main_{index}`` and its
        ``tvm_ffi.func`` is named ``{fn.__name__}_{index}`` to avoid symbol
        collisions in the merged module.
        """

        # Step 1: Export  ---------------------------------------------------
        gm, gs = torch._dynamo.export(
            fn, aten_graph=True, assume_static_by_default=True
        )(*args, **kwargs)
        gm(*args, **kwargs)  # Warm-up

        # Step 2: Import FX -> MLIR  ----------------------------------------
        with apply_patch():
            importer: FxImporter = FxImporter(context=ctx)
            main_func_name: Final[str] = f"main_{index}"
            main_func = importer.import_stateless_graph(
                gm.graph, func_name=main_func_name
            )
            module: ir.Module = importer.module

        torch._dynamo.reset()

        # Step 3: Wrap with tvm_ffi.func  ----------------------------------
        tvm_ffi_name: Final[str] = f"{fn.__name__}_{index}"
        arg_attrs: ir.ArrayAttr = parse_guards(gs).build(
            [*inspect.signature(fn).parameters.keys()], ctx
        )

        with ir.InsertionPoint(module.body), main_func.operation.location:
            ffi_func: tvm_ffi_d.FuncOp = tvm_ffi_d.func(
                tvm_ffi_name,
                ir.TypeAttr.get(main_func.type),
                arg_attrs=arg_attrs,
            )
            entry_block: ir.Block = ir.Block.create_at_start(
                ffi_func.body, main_func.type.inputs
            )
            with ir.InsertionPoint(entry_block):
                call_op = func.CallOp(
                    main_func.type.results,
                    main_func_name,
                    entry_block.arguments,
                )
                tvm_ffi_d.return_(call_op)

        return module

    # ------------------------------------------------------------------ #
    # Internal: module merging
    # ------------------------------------------------------------------ #

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
    # Internal: TVMFFIAny helpers
    # ------------------------------------------------------------------ #

    def _build_tvmffi_any(
        self,
        index: int,
        payload: ir.Value,
    ) -> ir.Value:
        """Build a TVMFFIAny struct value ``!llvm.struct<(i32, i32, i64)>``.

        Returns an SSA value with fields::

            {type_index=index, zero_padding=0, payload}

        ``payload`` must be an ``i64`` value (e.g. a ``zero_i64`` constant,
        or the result of ``ptrtoint``, …).
        """
        i32_ty: ir.Type = ir.IntegerType.get_signless(32)
        i64_ty: ir.Type = ir.IntegerType.get_signless(64)
        undef: ir.Value = llvm.mlir_undef(
            res=llvm.StructType.get_literal([i32_ty, i32_ty, i64_ty], context=self.ctx),
        )
        with_index: ir.Value = llvm.insertvalue(
            container=undef,
            value=llvm.mlir_constant(
                value=ir.IntegerAttr.get(
                    i32_ty,
                    index,
                ),
            ),
            position=ir.DenseI64ArrayAttr.get([0]),
        )
        with_padding: ir.Value = llvm.insertvalue(
            container=with_index,
            value=llvm.mlir_constant(
                value=ir.IntegerAttr.get(
                    i32_ty,
                    0,
                ),
            ),
            position=ir.DenseI64ArrayAttr.get([1]),
        )
        result: ir.Value = llvm.insertvalue(
            container=with_padding,
            value=payload,
            position=ir.DenseI64ArrayAttr.get([2]),
        )
        return result

    def _fill_tvmffi_any(
        self,
        slot: ir.Value,
        index: int,
        payload: ir.Value,
    ) -> None:
        """Store a TVMFFIAny into the alloca'd *slot*.

        Builds a complete ``!llvm.struct<(i32, i32, i64)>`` via
        ``_build_tvmffi_any`` and stores it in one shot.
        """
        struct_val: ir.Value = self._build_tvmffi_any(index, payload)
        llvm.store(value=struct_val, addr=slot)

    def _alloca_tvmffi_any(
        self,
        index: int,
        payload: ir.Value,
    ) -> ir.Value:
        """Allocate a TVMFFIAny slot, fill it, and return the pointer.

        A convenience that combines ``llvm.alloca`` + ``_fill_tvmffi_any``::

            %slot = llvm.alloca %any_ty
            _fill_tvmffi_any(%slot, index, payload)
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
        self._fill_tvmffi_any(slot, index, payload)
        return slot

    # ------------------------------------------------------------------ #
    # Internal: LLVM dispatcher
    # ------------------------------------------------------------------ #

    def _build_llvm_dispatcher(self, module: ir.Module) -> None:
        """After the pipeline has lowered every sub-module to LLVM IR, add a
        new ``llvm.func`` that tries each sub-function in sequence.

        The dispatcher ABI matches the TVM PackedCFunc convention::

            i32 (ptr, ptr, i32, ptr)

        Each sub-function writes its result into *ret_ptr*.  The dispatcher
        inspects the type_index field (field 0) of the resulting TVMFFIAny:
        if it matches the registered type index of ``trident.ffi.Exception``
        the dispatcher tries the next specialization.  If all fail, the
        last Exception is left in *ret_ptr* and the dispatcher returns 0
        (success), letting the Python caller see the Exception ObjectRef.

        A real error (sub-function returned -1) means the FFI error is
        already set; the dispatcher simply returns -1.
        """
        n: int = len(self._sub_modules)
        symbol: str = f"__tvm_ffi_{self.fn.__name__}"

        # ── Types ────────────────────────────────────────────────────────
        with self.ctx:
            i32_type: ir.IntegerType = ir.IntegerType.get_signless(32)
            i64_type: ir.IntegerType = ir.IntegerType.get_signless(64)
            ptr_type: ir.Type = llvm.PointerType.get()

        with ir.InsertionPoint(module.body), module.operation.location:
            # ── Declare external C API functions (only when missing) ─
            sym_tab: ir.SymbolTable = ir.SymbolTable(module.operation)
            needed_funcs: List[Tuple[str, str]] = [
                ("TVMFFIFunctionGetGlobal", "!llvm.func<i32 (!llvm.ptr, !llvm.ptr)>"),
                (
                    "TVMFFIFunctionCall",
                    "!llvm.func<i32 (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr)>",
                ),
                ("TVMFFIObjectDecRef", "!llvm.func<i32 (!llvm.ptr)>"),
            ]
            for fname, ftype_str in needed_funcs:
                if fname not in sym_tab:
                    llvm.func(
                        sym_name=fname,
                        function_type=ir.TypeAttr.get(
                            ir.Type.parse(ftype_str, self.ctx),
                        ),
                    )

            # ── Create the dispatcher function ────────────────────────
            dispatcher: llvm.FuncOp = llvm.func(
                sym_name=symbol,
                function_type=ir.TypeAttr.get(
                    ir.Type.parse(
                        "!llvm.func<i32 (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr)>",
                        self.ctx,
                    )
                ),
            )

            # Entry block: receives ABI arguments.
            entry_block: ir.Block = ir.Block.create_at_start(
                dispatcher.body,
                arg_types=[ptr_type, ptr_type, i32_type, ptr_type],
            )
            in_ptr, out_ptr, num_args, ret_ptr = entry_block.arguments

            # ── Forward-declared blocks ─────────────────────────────
            error_block: ir.Block = ir.Block.create_after(entry_block)
            done_block: ir.Block = ir.Block.create_after(error_block)
            exit_block: ir.Block = ir.Block.create_after(done_block, i32_type)
            (exit_ret_val,) = exit_block.arguments

            # ── Common types ──────────────────────────────────────────
            any_ty: ir.Type = llvm.StructType.get_literal(
                [i32_type, i32_type, i64_type],
                context=self.ctx,
            )
            byte_array_ty: ir.Type = llvm.StructType.get_literal(
                [ptr_type, i64_type],
                context=self.ctx,
            )

            with ir.InsertionPoint(entry_block):
                # SSA constants (dominate all blocks)
                one_i64: ir.Value = llvm.mlir_constant(
                    value=ir.IntegerAttr.get(i64_type, 1),
                )
                zero_i32: ir.Value = llvm.mlir_constant(
                    value=ir.IntegerAttr.get(i32_type, 0),
                )
                zero_i64: ir.Value = llvm.mlir_constant(
                    value=ir.IntegerAttr.get(i64_type, 0),
                )
                one_i32: ir.Value = llvm.mlir_constant(
                    value=ir.IntegerAttr.get(i32_type, 1),
                )

                # ── Module-level null-terminated C string for Exception ─
                gme_kind: Final[str] = "GuardMatchException\0"
                with ir.InsertionPoint(module.body):
                    llvm.mlir_global(
                        global_type=ir.TypeAttr.get(
                            ir.Type.parse(
                                f"!llvm.array<{len(gme_kind)} x i8>",
                                self.ctx,
                            )
                        ),
                        sym_name="__trident_const_GuardMatchException",
                        linkage=ir.Attribute.parse(
                            "#llvm.linkage<internal>",
                            self.ctx,
                        ),
                        constant=True,
                        value=ir.StringAttr.get(gme_kind),
                    )

                # ── Pre-fetch handle: trident.ffi.GetExceptionIndex ──
                handles: Dict[str, ir.Value] = {}
                for fname in [
                    "trident.ffi.GetExceptionIndex",
                    "trident.ffi.Exception",
                ]:
                    gsym: str = f"__trident_const_{fname}"
                    # mlir_global string constant (module level)
                    with ir.InsertionPoint(module.body):
                        name_len: Final[int] = len(fname)
                        llvm.mlir_global(
                            global_type=ir.TypeAttr.get(
                                ir.Type.parse(
                                    f"!llvm.array<{name_len} x i8>",
                                    self.ctx,
                                )
                            ),
                            sym_name=gsym,
                            linkage=ir.Attribute.parse(
                                "#llvm.linkage<internal>",
                                self.ctx,
                            ),
                            constant=True,
                            value=ir.StringAttr.get(fname),
                        )
                    # Pre-fetch handle in entry block
                    name_ptr: ir.Value = llvm.AddressOfOp(
                        res=ptr_type,
                        global_name=ir.FlatSymbolRefAttr.get(gsym),
                    )
                    name_slot: ir.Value = llvm.alloca(
                        res=ptr_type,
                        array_size=one_i64,
                        elem_type=ir.TypeAttr.get(byte_array_ty),
                    )
                    llvm.store(
                        value=name_ptr,
                        addr=llvm.getelementptr(
                            res=ptr_type,
                            base=name_slot,
                            dynamic_indices=[],
                            raw_constant_indices=ir.DenseI32ArrayAttr.get([0, 0]),
                            elem_type=ir.TypeAttr.get(byte_array_ty),
                            no_wrap_flags=None,
                        ),
                    )
                    llvm.store(
                        value=llvm.mlir_constant(
                            value=ir.IntegerAttr.get(i64_type, len(fname)),
                        ),
                        addr=llvm.getelementptr(
                            res=ptr_type,
                            base=name_slot,
                            dynamic_indices=[],
                            raw_constant_indices=ir.DenseI32ArrayAttr.get([0, 1]),
                            elem_type=ir.TypeAttr.get(byte_array_ty),
                            no_wrap_flags=None,
                        ),
                    )
                    handle_slot: ir.Value = llvm.alloca(
                        res=ptr_type,
                        array_size=one_i64,
                        elem_type=ir.TypeAttr.get(ptr_type),
                    )
                    llvm.call(
                        result=i32_type,
                        callee_operands=[name_slot, handle_slot],
                        op_bundle_operands=[],
                        op_bundle_sizes=[],
                        callee=ir.FlatSymbolRefAttr.get(
                            "TVMFFIFunctionGetGlobal",
                        ),
                    )
                    handles[fname] = llvm.load(res=ptr_type, addr=handle_slot)

                # ── Get ExcIdx = trident.ffi.GetExceptionIndex() ─────
                exc_idx_slot: ir.Value = self._alloca_tvmffi_any(0, zero_i64)
                llvm.call(
                    result=i32_type,
                    callee_operands=[
                        handles["trident.ffi.GetExceptionIndex"],
                        llvm.inttoptr(
                            res=ptr_type,
                            arg=zero_i64,
                        ),
                        zero_i32,
                        exc_idx_slot,
                    ],
                    op_bundle_operands=[],
                    op_bundle_sizes=[],
                    callee=ir.FlatSymbolRefAttr.get(
                        "TVMFFIFunctionCall",
                    ),
                )

                # ── Allocate local result slot (TVMFFIAny) ──
                result_slot: ir.Value = self._alloca_tvmffi_any(0, zero_i64)

                # ── Construct Exception("GuardMatchException") via FFI ──
                # When n==0 the dispatcher returns this immediately,
                # signalling recompile.  When n>0 sub-functions
                # overwrite it on success (or leave their own
                # Exception on failure).

                # Load exc_idx from GetExceptionIndex result for comparison
                exc_idx_i32: ir.Value = llvm.trunc(
                    res=i32_type,
                    arg=llvm.load(
                        res=i64_type,
                        addr=llvm.getelementptr(
                            res=ptr_type,
                            base=exc_idx_slot,
                            dynamic_indices=[],
                            raw_constant_indices=ir.DenseI32ArrayAttr.get([0, 2]),
                            elem_type=ir.TypeAttr.get(any_ty),
                            no_wrap_flags=None,
                        ),
                    ),
                    overflow_flags=ir.Attribute.parse("#llvm.overflow<none>", self.ctx),
                )

                # Build TVMFFIAny {kTVMFFIRawStr=8, padding=0, v_c_str=&"GuardMatchException\0"}
                # AnyView::type_index can be kTVMFFIRawStr (8) —
                # the std::string fallback chain accepts it via const char*,
                # no heap allocation needed.
                # v_c_str = pointer to the null-terminated global string
                args_slot: ir.Value = self._alloca_tvmffi_any(
                    8,
                    llvm.ptrtoint(
                        res=i64_type,
                        arg=llvm.AddressOfOp(
                            res=ptr_type,
                            global_name=ir.FlatSymbolRefAttr.get(
                                "__trident_const_GuardMatchException"
                            ),
                        ),
                    ),
                )

                # Call trident.ffi.Exception(kind="GuardMatchException") via RawStr
                llvm.call(
                    result=i32_type,
                    callee_operands=[
                        handles["trident.ffi.Exception"],
                        args_slot,
                        one_i32,
                        result_slot,
                    ],
                    op_bundle_operands=[],
                    op_bundle_sizes=[],
                    callee=ir.FlatSymbolRefAttr.get("TVMFFIFunctionCall"),
                )

                # ── Streaming: try_i → check_i → try_{i+1} ─────
            prev_block: ir.Block = entry_block

            for i in range(n):
                sub_symbol: Final[str] = f"{symbol}_{i}"

                # Create try_i + check_i
                try_blk: ir.Block = ir.Block.create_after(prev_block)
                check_blk: ir.Block = ir.Block.create_after(try_blk)

                # Terminator for prev_block → try_i / done
                with ir.InsertionPoint(prev_block):
                    llvm.cond_br(
                        condition=llvm.icmp(
                            predicate=llvm.ICmpPredicate.eq,
                            lhs=llvm.load(
                                res=i32_type,
                                addr=llvm.getelementptr(
                                    res=ptr_type,
                                    base=result_slot,
                                    dynamic_indices=[],
                                    raw_constant_indices=ir.DenseI32ArrayAttr.get(
                                        [0, 0]
                                    ),
                                    elem_type=ir.TypeAttr.get(any_ty),
                                    no_wrap_flags=None,
                                ),
                            ),
                            rhs=exc_idx_i32,
                        ),
                        true_dest_operands=[],
                        false_dest_operands=[],
                        true_dest=try_blk,
                        false_dest=done_block,
                    )

                with ir.InsertionPoint(try_blk):
                    # Release the previous exception object to avoid leak
                    llvm.call(
                        result=i32_type,
                        callee_operands=[
                            llvm.inttoptr(
                                res=ptr_type,
                                arg=llvm.load(
                                    res=i64_type,
                                    addr=llvm.getelementptr(
                                        res=ptr_type,
                                        base=result_slot,
                                        dynamic_indices=[],
                                        raw_constant_indices=ir.DenseI32ArrayAttr.get(
                                            [0, 2]
                                        ),
                                        elem_type=ir.TypeAttr.get(any_ty),
                                        no_wrap_flags=None,
                                    ),
                                ),
                            )
                        ],
                        op_bundle_operands=[],
                        op_bundle_sizes=[],
                        callee=ir.FlatSymbolRefAttr.get("TVMFFIObjectDecRef"),
                    )
                    llvm.cond_br(
                        condition=llvm.icmp(
                            predicate=llvm.ICmpPredicate.eq,
                            lhs=llvm.call(
                                result=i32_type,
                                callee_operands=[
                                    in_ptr,
                                    out_ptr,
                                    num_args,
                                    result_slot,
                                ],
                                op_bundle_operands=[],
                                op_bundle_sizes=[],
                                callee=ir.FlatSymbolRefAttr.get(sub_symbol),
                            ),
                            rhs=zero_i32,
                        ),
                        true_dest_operands=[],
                        false_dest_operands=[],
                        true_dest=check_blk,
                        false_dest=error_block,
                    )

                prev_block = check_blk

            # Terminator for last prev_block → done_block
            with ir.InsertionPoint(prev_block):
                llvm.br(dest_operands=[], dest=done_block)

            # ── error block (shared): real error → -1 ────────────────
            with ir.InsertionPoint(error_block):
                llvm.br(
                    dest_operands=[
                        llvm.mlir_constant(
                            value=ir.IntegerAttr.get(i32_type, -1),
                        )
                    ],
                    dest=exit_block,
                )

            # ── done block ────────────────────────────────────────────
            with ir.InsertionPoint(done_block):
                # Copy result_slot → *ret_ptr (entire TVMFFIAny struct at once)
                llvm.store(
                    value=llvm.load(res=any_ty, addr=result_slot),
                    addr=ret_ptr,
                )
                llvm.br(dest_operands=[zero_i32], dest=exit_block)

            # ── exit block (shared): dec-ref + return ─────────────
            with ir.InsertionPoint(exit_block):
                for h in handles.values():
                    llvm.call(
                        result=i32_type,
                        callee_operands=[h],
                        op_bundle_operands=[],
                        op_bundle_sizes=[],
                        callee=ir.FlatSymbolRefAttr.get(
                            "TVMFFIObjectDecRef",
                        ),
                    )
                llvm.return_(arg=exit_ret_val)
