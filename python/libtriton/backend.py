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


class LibTritonGraphModule(object):
    """Compiles a torch function via Torch-MLIR and wraps with tvm_ffi.

    Maintains a list of sub-modules, one per unique guard specialization
    encountered at call time.  On each ``compile()`` call the class builds
    a fresh combined module that contains every sub-module plus an LLVM-level
    dispatcher.  The dispatcher tries each sub-module's compiled function in
    order – the first one that returns 0 (guard match) wins.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Callable[..., Any]] = fn
        self.ctx: Final[ir.Context] = ir.Context()
        register_all_dialects(self.ctx)
        register_all_passes()
        self._sub_modules: List[ir.Module] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(self, *args: Any) -> Any:
        executor = self.compile(*args)
        return executor(*args)

    def compile(self, *args: Any) -> Callable[..., Any]:
        fn: tvm_ffi.Function = self._compile_and_link(*args)

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
    # Internal: orchestration
    # ------------------------------------------------------------------ #

    def _compile_and_link(self, *args: Any) -> tvm_ffi.Function:
        """Build a new sub-module, merge all sub-modules, lower, add an
        LLVM-level dispatcher, and JIT-compile."""

        # 1. Build a new sub-module for the current arguments.
        sub_mod: ir.Module = self._build_sub_module(
            self.fn, self.ctx, len(self._sub_modules), args
        )
        self._sub_modules.append(sub_mod)

        # 2. Merge all sub-modules into one.
        combined: ir.Module = self._build_combined_module()

        # 3. Lower Torch / TVM-FFI → LLVM.
        with self.ctx:
            passmanager.PassManager.parse("builtin.module(torch-to-llvm-pipeline)").run(
                combined.operation
            )

        # 4. Build the LLVM dispatcher that tries each sub-function in order.
        self._build_llvm_dispatcher(combined)

        # 5. JIT-compile and wrap.
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

        return tvm_ffi.Function.__from_mlir_packed_safe_call__(
            ptr, keep_alive_object=engine
        )

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
        """Merge all sub-modules into a single ``ir.Module`` via
        ``copy_symbols_and_merge_into``.

        Module-level attributes (such as ``gpu.container_module``) are
        propagated from the source modules to the combined module *before*
        the merge call to satisfy intra-module verification constraints.
        """
        combined: ir.Module = ir.Module.create(loc=ir.Location.unknown(self.ctx))
        with self.ctx, ir.Location.unknown(self.ctx):
            for sub_mod in self._sub_modules:
                for attr_name in sub_mod.operation.attributes:
                    if (
                        attr_name not in combined.operation.attributes
                        and not attr_name.startswith("sym_name")
                    ):
                        combined.operation.attributes[attr_name] = (
                            sub_mod.operation.attributes[attr_name]
                        )

                transform.interpreter.copy_symbols_and_merge_into(
                    combined.operation, sub_mod.operation
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

        The dispatcher returns the **first** sub-function result that equals
        zero (success).  If all sub-functions return non-zero, the dispatcher
        returns -1.
        """
        n: int = len(self._sub_modules)
        symbol: str = f"__tvm_ffi_{self.fn.__name__}"

        # ── Types ────────────────────────────────────────────────────────
        with self.ctx:
            i32_type: ir.IntegerType = ir.IntegerType.get_signless(32)
            ptr_type = llvm.PointerType.get()
            func_type: ir.Type = ir.Type.parse(
                "!llvm.func<i32 (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr)>",
                self.ctx,
            )

        with ir.InsertionPoint(module.body), ir.Location.unknown(self.ctx):
            # ── Create the dispatcher function ────────────────────────
            dispatcher = llvm.func(
                sym_name=symbol,
                function_type=ir.TypeAttr.get(func_type),
            )
            region: ir.Region = dispatcher.body

            # Entry block: receives ABI arguments.
            entry_block: ir.Block = ir.Block.create_at_start(
                region,
                arg_types=[ptr_type, ptr_type, i32_type, ptr_type],
            )

            # ── Pre-create all blocks in the correct MLIR order ───────
            # (use create_after to chain them: entry → try_0 → … → try_N → fail → done)
            prev_block: ir.Block = entry_block
            try_blocks: List[ir.Block] = []
            for _ in range(n):
                blk = ir.Block.create_after(prev_block)
                try_blocks.append(blk)
                prev_block = blk

            fail_block: ir.Block = ir.Block.create_after(prev_block)
            done_block: ir.Block = ir.Block.create_after(fail_block, i32_type)

            # ── entry → try_0 ─────────────────────────────────────────
            with ir.InsertionPoint(entry_block):
                llvm.br(dest_operands=[], dest=try_blocks[0])

            # ── Build each try_i block ─────────────────────────────────
            for i, blk in enumerate(try_blocks):
                sub_symbol: str = f"{symbol}_{i}"

                with ir.InsertionPoint(blk):
                    # Call sub-module function – returns i32 result directly.
                    ret = llvm.call(
                        result=i32_type,
                        callee_operands=entry_block.arguments,
                        op_bundle_operands=[],
                        op_bundle_sizes=[],
                        callee=ir.FlatSymbolRefAttr.get(sub_symbol),
                    )
                    zero = llvm.mlir_constant(
                        value=ir.IntegerAttr.get(i32_type, 0),
                    )
                    ok = llvm.icmp(
                        predicate=llvm.ICmpPredicate.eq,
                        lhs=ret,
                        rhs=zero,
                    )

                    # Conditional branch: success → done, else → next try / fail
                    next_blk: ir.Block = try_blocks[i + 1] if i < n - 1 else fail_block
                    llvm.cond_br(
                        condition=ok,
                        true_dest_operands=[ret],
                        false_dest_operands=[],
                        true_dest=done_block,
                        false_dest=next_blk,
                    )

            # ── fail block ────────────────────────────────────────────
            with ir.InsertionPoint(fail_block):
                neg_one = llvm.mlir_constant(
                    value=ir.IntegerAttr.get(i32_type, -1),
                )
                llvm.return_(arg=neg_one)

            # ── done block ────────────────────────────────────────────
            with ir.InsertionPoint(done_block):
                llvm.return_(arg=done_block.arguments[0])
