from __future__ import annotations
from typing import Any, Callable, Final, List, Tuple

import inspect
import torch
import tvm_ffi

from libtriton._C.libtriton_core import (
    capi_utils,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from libtriton._C.libtriton_core.dialects import (
    func,
    llvm,
    transform,
    tvm_ffi as tvm_ffi_d,
)
from libtriton._C.libtriton_core.execution_engine import ExecutionEngine
from .guards import parse_guards
from .importer import LibTritonFxImporter
from .error import GuardMatchException


class LibTritonGraphModule(object):
    """Compiles a torch function via Torch-MLIR and wraps with tvm_ffi.

    Maintains a list of sub-modules, one per unique guard specialization
    encountered at call time.  On each ``compile()`` call the class builds
    a fresh combined module that contains every sub-module plus an LLVM-level
    dispatcher.  The dispatcher tries each sub-module's compiled function in
    order - the first one that returns 0 (guard match) wins.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Callable[..., Any]] = fn
        self.ctx: Final[ir.Context] = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()
        self._sub_modules: List[ir.Module] = []
        self.executor: Callable[..., Any] = self.stub_compile()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(self, *args: Any) -> Any:
        max_compiles: int = len(self._sub_modules) + 10
        while True:
            try:
                return self.executor(*args)
            except GuardMatchException:
                if len(self._sub_modules) >= max_compiles:
                    raise RuntimeError(
                        f"recompilation limit ({max_compiles}) exceeded "
                        f"without finding a matching specialization"
                    ) from None
                self.compile(*args)

    def compile(self, *args: Any) -> Any:
        """Build a new sub-module for *args* and rebuild the combined
        module + dispatcher.  Called automatically from ``__call__`` when a
        ``GuardMatchException`` is raised."""
        # 1. Build a new sub-module for the current arguments.
        sub_mod: ir.Module = self._build_sub_module(
            self.fn, self.ctx, len(self._sub_modules), args
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
        # ── n == 0: no sub-modules yet → stub that always triggers a
        # recompile via GuardMatchException.
        if len(self._sub_modules) == 0:

            def _stub(*_args: Any) -> Any:
                raise GuardMatchException(
                    "no suitable specialization compiled yet",
                )

            return _stub

        # 1. Merge all sub-modules into one (each is cloned to avoid
        #    repeated-merging hangs).
        combined: ir.Module = self._build_combined_module()

        # 2. Lower Torch / TVM-FFI → LLVM.
        with self.ctx:
            passmanager.PassManager.parse(
                "builtin.module(torch-to-llvm-pipeline)",
            ).run(combined.operation)

        # 3. Build the LLVM dispatcher that tries each sub-function.
        self._build_llvm_dispatcher(combined)

        # 4. JIT-compile and wrap.
        engine: ExecutionEngine = ExecutionEngine(
            combined,
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

        def executor(*inner_args: Any) -> Any:
            def canonicalize(v: Any) -> Any:
                if isinstance(v, (list, tuple, tvm_ffi.Array)):
                    return type(v)(canonicalize(x) for x in v)
                elif hasattr(v, "__dlpack__") and not isinstance(v, torch.Tensor):
                    return torch.from_dlpack(v)
                else:
                    return v

            return canonicalize(fn(*inner_args))

        return executor

    # ------------------------------------------------------------------ #
    # Internal: sub-module construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_sub_module(
        fn: Callable[..., Any],
        ctx: ir.Context,
        index: int,
        args: Tuple[Any, ...],
    ) -> ir.Module:
        """Export → import → wrap a single sub-module for *args*.

        Each sub-module's ``func.func`` is named ``main_{index}`` and its
        ``tvm_ffi.func`` is named ``{fn.__name__}_{index}`` to avoid symbol
        collisions in the merged module.
        """

        # Step 1: Export  ---------------------------------------------------
        gm, _guards = torch._dynamo.export(
            fn, aten_graph=True, assume_static_by_default=True
        )(*args)
        gm(*args)  # Warm-up

        # Step 2: Import FX → MLIR  ----------------------------------------
        importer: LibTritonFxImporter = LibTritonFxImporter(context=ctx)
        main_func_name: str = f"main_{index}"
        main_func = importer.import_stateless_graph(gm.graph, func_name=main_func_name)
        module: ir.Module = importer.module

        # Step 3: Wrap with tvm_ffi.func  ----------------------------------
        tvm_ffi_name: str = f"{fn.__name__}_{index}"
        arg_attrs: ir.ArrayAttr = parse_guards(_guards).build(
            [*inspect.signature(fn).parameters.keys()], ctx
        )

        with ir.InsertionPoint(module.body), ir.Location.unknown(ctx):
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

        Each sub-module is cloned via round-trip through its MLIR text
        representation before merging so that ``copy_symbols_and_merge_into``
        always sees a fresh source module.  This avoids hangs caused by
        repeated merging of the same module objects.
        """
        combined: ir.Module = ir.Module.create(
            loc=ir.Location.unknown(self.ctx),
        )
        with self.ctx, ir.Location.unknown(self.ctx):
            for sub_mod in self._sub_modules:
                # Clone to get a fresh source for copy_symbols_and_merge_into.
                clone: ir.Module = ir.Module.parse(
                    str(sub_mod),
                    self.ctx,
                )
                for attr_name in clone.operation.attributes:
                    if (
                        attr_name not in combined.operation.attributes
                        and not attr_name.startswith("sym_name")
                    ):
                        combined.operation.attributes[attr_name] = (
                            clone.operation.attributes[attr_name]
                        )

                transform.interpreter.copy_symbols_and_merge_into(
                    combined.operation,
                    clone.operation,
                )
        return combined

    # ------------------------------------------------------------------ #
    # Internal: LLVM dispatcher
    # ------------------------------------------------------------------ #

    def _build_llvm_dispatcher(self, module: ir.Module) -> None:
        """After the pipeline has lowered every sub-module to LLVM IR, add a
        new ``llvm.func`` that tries each sub-function in sequence.

        The dispatcher ABI matches the TVM PackedCFunc convention::

            i32 (ptr, ptr, i32, ptr)

        The dispatcher checks both the return value and the raised error.
        If a sub-function returns 0 (success) the dispatcher returns that
        result.  If the sub-function returns non-zero and the raised error
        kind is ``GuardMatchException``, the dispatcher tries the next
        sub-function.  Any other error is propagated immediately.
        If all sub-functions fail, the dispatcher returns -1.
        """
        n: int = len(self._sub_modules)
        symbol: str = f"__tvm_ffi_{self.fn.__name__}"

        # ── Types ────────────────────────────────────────────────────────
        with self.ctx:
            i1_type: ir.IntegerType = ir.IntegerType.get_signless(1)
            i8_type: ir.IntegerType = ir.IntegerType.get_signless(8)
            i32_type: ir.IntegerType = ir.IntegerType.get_signless(32)
            i64_type: ir.IntegerType = ir.IntegerType.get_signless(64)
            ptr_type: ir.Type = llvm.PointerType.get()

        with ir.InsertionPoint(module.body), module.operation.location:
            # ── Declare external C API functions ──────────────────────
            error_api_type: ir.Type = ir.Type.parse(
                "!llvm.func<void (!llvm.ptr)>",
                self.ctx,
            )
            llvm.func(
                sym_name="TVMFFIErrorMoveFromRaised",
                function_type=ir.TypeAttr.get(error_api_type),
            )
            llvm.func(
                sym_name="TVMFFIErrorSetRaised",
                function_type=ir.TypeAttr.get(error_api_type),
            )

            # ── Global string constant for "GuardMatchException" ──────
            guard_str: str = "GuardMatchException"
            guard_str_len: int = len(guard_str)
            llvm.mlir_global(
                global_type=ir.TypeAttr.get(
                    ir.Type.parse(
                        f"!llvm.array<{guard_str_len} x i8>",
                        self.ctx,
                    )
                ),
                sym_name="__guard_match_exception_str",
                linkage=ir.Attribute.parse(
                    "#llvm.linkage<internal>",
                    self.ctx,
                ),
                constant=True,
                value=ir.StringAttr.get(guard_str),
            )

            # ── Declare memcmp from libc ──────────────────────────────
            llvm.func(
                sym_name="memcmp",
                function_type=ir.TypeAttr.get(
                    ir.Type.parse(
                        "!llvm.func<i32 (!llvm.ptr, !llvm.ptr, i64)>",
                        self.ctx,
                    )
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

            # ── Pre-create all blocks ─────────────────────────────────
            # Layout: entry →
            #   (try_0, try_0_no_err, try_0_has_err, try_0_merge) →
            #   … →
            #   (try_N, …, try_N_merge) → fail → done
            prev_block: ir.Block = entry_block
            blocks: List[Tuple[ir.Block, ir.Block, ir.Block, ir.Block]] = []

            for _ in range(n):
                blk: ir.Block = ir.Block.create_after(prev_block)
                blk_no: ir.Block = ir.Block.create_after(blk)
                blk_err: ir.Block = ir.Block.create_after(blk_no)
                # merge block takes i1 (error_ok flag)
                blk_merge: ir.Block = ir.Block.create_after(blk_err, i1_type)
                prev_block = blk_merge
                blocks.append((blk, blk_no, blk_err, blk_merge))

            fail_block: ir.Block = ir.Block.create_after(prev_block)
            done_block: ir.Block = ir.Block.create_after(
                fail_block,
                i32_type,
            )

            # ── entry → try_0 ─────────────────────────────────────────
            with ir.InsertionPoint(entry_block):
                [(try0, _, _, _), *_] = blocks
                llvm.br(dest_operands=[], dest=try0)

            # ── Build each try_i group ─────────────────────────────────
            for i, (try_blk, no_err_blk, has_err_blk, merge_blk) in enumerate(blocks):
                sub_symbol: str = f"{symbol}_{i}"

                # ── try_i: call sub-function & fetch error ────────
                with ir.InsertionPoint(try_blk):
                    # Call sub-module function → i32 result.
                    ret: ir.Value = llvm.call(
                        result=i32_type,
                        callee_operands=entry_block.arguments,
                        op_bundle_operands=[],
                        op_bundle_sizes=[],
                        callee=ir.FlatSymbolRefAttr.get(sub_symbol),
                    )

                    # Allocate space for the error handle.
                    one_i32: ir.Value = llvm.mlir_constant(
                        value=ir.IntegerAttr.get(i32_type, 1),
                    )
                    error_ptr: ir.Value = llvm.alloca(
                        res=ptr_type,
                        array_size=one_i32,
                        elem_type=ir.TypeAttr.get(ptr_type),
                    )

                    # TVMFFIErrorMoveFromRaised(&error_ptr)
                    llvm.call(
                        result=None,
                        callee_operands=[error_ptr],
                        op_bundle_operands=[],
                        op_bundle_sizes=[],
                        callee=ir.FlatSymbolRefAttr.get(
                            "TVMFFIErrorMoveFromRaised",
                        ),
                    )

                    # Load error handle.
                    error: ir.Value = llvm.load(res=ptr_type, addr=error_ptr)

                    # Check if error is NULL.
                    null_ptr: ir.Value = llvm.mlir_zero(res=ptr_type)
                    is_null: ir.Value = llvm.icmp(
                        predicate=llvm.ICmpPredicate.eq,
                        lhs=error,
                        rhs=null_ptr,
                    )

                    # Branch: null → no_error, non-null → has_error.
                    llvm.cond_br(
                        condition=is_null,
                        true_dest_operands=[],
                        false_dest_operands=[],
                        true_dest=no_err_blk,
                        false_dest=has_err_blk,
                    )

                # ── try_i_no_error: error is null → ok from ret ────
                with ir.InsertionPoint(no_err_blk):
                    true_i1: ir.Value = llvm.mlir_constant(
                        value=ir.IntegerAttr.get(i1_type, 1),
                    )
                    llvm.br(dest_operands=[true_i1], dest=merge_blk)

                # ── try_i_has_error: check kind ────────────────────
                with ir.InsertionPoint(has_err_blk):
                    # Reload error (still on stack from alloca).
                    err_val: ir.Value = llvm.load(res=ptr_type, addr=error_ptr)

                    # Put the error back (always).
                    llvm.call(
                        result=None,
                        callee_operands=[err_val],
                        op_bundle_operands=[],
                        op_bundle_sizes=[],
                        callee=ir.FlatSymbolRefAttr.get(
                            "TVMFFIErrorSetRaised",
                        ),
                    )

                    # ── Get error cell (offset sizeof(TVMFFIObject)=24)
                    cell_ptr: ir.Value = llvm.getelementptr(
                        res=ptr_type,
                        base=err_val,
                        dynamic_indices=[],
                        raw_constant_indices=ir.DenseI32ArrayAttr.get(
                            [24],
                        ),
                        elem_type=ir.TypeAttr.get(i8_type),
                        no_wrap_flags=None,
                    )

                    # kind.data (at offset 0 in TVMFFIByteArray)
                    kind_data: ir.Value = llvm.load(res=ptr_type, addr=cell_ptr)

                    # kind.size (at offset 8 in TVMFFIByteArray)
                    size_ptr: ir.Value = llvm.getelementptr(
                        res=ptr_type,
                        base=cell_ptr,
                        dynamic_indices=[],
                        raw_constant_indices=ir.DenseI32ArrayAttr.get(
                            [8],
                        ),
                        elem_type=ir.TypeAttr.get(i8_type),
                        no_wrap_flags=None,
                    )
                    kind_size: ir.Value = llvm.load(res=i64_type, addr=size_ptr)

                    # Compare length against guard string length.
                    guard_len: ir.Value = llvm.mlir_constant(
                        value=ir.IntegerAttr.get(i64_type, guard_str_len),
                    )
                    len_ok: ir.Value = llvm.icmp(
                        predicate=llvm.ICmpPredicate.eq,
                        lhs=kind_size,
                        rhs=guard_len,
                    )

                    # ── Compare via memcmp ─────────────────────────
                    # memcmp(guard_ptr, kind_data, guard_str_len) == 0
                    guard_ptr: ir.Value = llvm.AddressOfOp(
                        res=ptr_type,
                        global_name=ir.FlatSymbolRefAttr.get(
                            "__guard_match_exception_str",
                        ),
                    )
                    cmp_result: ir.Value = llvm.call(
                        result=i32_type,
                        callee_operands=[
                            guard_ptr,
                            kind_data,
                            llvm.mlir_constant(
                                value=ir.IntegerAttr.get(i64_type, guard_str_len),
                            ),
                        ],
                        op_bundle_operands=[],
                        op_bundle_sizes=[],
                        callee=ir.FlatSymbolRefAttr.get("memcmp"),
                    )
                    zero_i32: ir.Value = llvm.mlir_constant(
                        value=ir.IntegerAttr.get(i32_type, 0),
                    )
                    str_ok: ir.Value = llvm.icmp(
                        predicate=llvm.ICmpPredicate.eq,
                        lhs=cmp_result,
                        rhs=zero_i32,
                    )

                    # is_guard_match = len_ok AND str_ok
                    false_i1: ir.Value = llvm.mlir_constant(
                        value=ir.IntegerAttr.get(i1_type, 0),
                    )
                    is_guard: ir.Value = llvm.select(
                        condition=len_ok,
                        true_value=str_ok,
                        false_value=false_i1,
                    )

                    # error_ok = NOT is_guard
                    error_ok_i1: ir.Value = llvm.icmp(
                        predicate=llvm.ICmpPredicate.eq,
                        lhs=is_guard,
                        rhs=false_i1,
                    )

                    # Branch to merge with error_ok flag.
                    llvm.br(
                        dest_operands=[error_ok_i1],
                        dest=merge_blk,
                    )

                # ── try_i_merge: combine ret & error checks ────────
                with ir.InsertionPoint(merge_blk):
                    [error_ok] = merge_blk.arguments

                    # ok = (ret == 0) OR error_ok
                    zero: ir.Value = llvm.mlir_constant(
                        value=ir.IntegerAttr.get(i32_type, 0),
                    )
                    ret_ok_i1: ir.Value = llvm.icmp(
                        predicate=llvm.ICmpPredicate.eq,
                        lhs=ret,
                        rhs=zero,
                    )
                    ok: ir.Value = llvm.or_(ret_ok_i1, error_ok)

                    # Conditional branch: ok → done, else → next / fail
                    (next_blk, _, _, _) = (
                        blocks[i + 1] if i < n - 1 else (fail_block, None, None, None)
                    )
                    llvm.cond_br(
                        condition=ok,
                        true_dest_operands=[ret],
                        false_dest_operands=[],
                        true_dest=done_block,
                        false_dest=next_blk,
                    )

            # ── fail block ────────────────────────────────────────────
            with ir.InsertionPoint(fail_block):
                neg_one: ir.Value = llvm.mlir_constant(
                    value=ir.IntegerAttr.get(i32_type, -1),
                )
                llvm.return_(arg=neg_one)

            # ── done block ────────────────────────────────────────────
            with ir.InsertionPoint(done_block):
                [arg] = done_block.arguments
                llvm.return_(arg=arg)
