# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Final

import torch
import torch.utils._pytree as pytree
import tvm_ffi
import tvm_ffi.utils
from torch.fx.experimental.proxy_tensor import make_fx

from trident import capi_utils
from trident.core import (
    _convert_torch_type_to_tvm_ffi_type,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from trident.core.dialects import (
    func,
    llvm,
    scf,
    torchext,
    transform,
)
from trident.core.dialects import (
    torch as torch_d,
)
from trident.core.dialects import (
    tvm_ffi as tvm_ffi_d,
)
from trident.core.execution_engine import ExecutionEngine
from trident.core.extras.fx_importer import FxImporter
from trident.ffi import Exception

from .guards import parse_guards
from .patch import apply_patch


class _InputTreeNode:
    """Static shape information for one exported input pytree node."""

    __slots__ = ("child_indices", "children", "keys", "leaf_index", "type")

    def __init__(
        self,
        type: ir.Type,
        children: list[_InputTreeNode] = [],  # noqa: B006
        keys: list[int | str] | None = None,
        leaf_index: int | None = None,
    ) -> None:
        children = [*children]
        if keys is None:
            keys = [i for i, _ in enumerate(children)]
        else:
            keys = [*keys]
        assert len(keys) == len(children), (
            "tree keys and children must have equal lengths"
        )
        assert len(set(keys)) == len(keys), "tree keys must be unique"
        self.type: Final[ir.Type] = type
        self.children: Final[list[_InputTreeNode]] = children
        self.keys: Final[list[int | str]] = keys
        self.child_indices: Final[dict[int | str, int]] = {
            key: index for index, key in enumerate(keys)
        }
        self.leaf_index: Final[int | None] = leaf_index


class _InputTreeMap:
    """Lazily materialized IR values indexed by pytree paths.

    Values are materialized on demand so structure guards can run before an
    element access forces the corresponding TVM FFI array access.
    Each path lookup gets a fresh map, keeping values local to the current IR
    region.  ``flatten`` uses the persistent map to share unpacks while
    reconstructing the main function arguments.
    """

    def __init__(
        self,
        root: _InputTreeNode,
        arguments: dict[str, ir.Value],
    ) -> None:
        self._root: Final[_InputTreeNode] = root
        self._arguments: Final[dict[str, ir.Value]] = arguments
        self._values: dict[str, ir.Value] = {
            self._path_key([name]): argument for name, argument in arguments.items()
        }

    @staticmethod
    def _path_key(path: list[int | str]) -> str:
        return repr(path)

    def _unpack(
        self,
        node: _InputTreeNode,
        path: list[int | str],
        values: dict[str, ir.Value],
    ) -> None:
        if not node.children:
            return
        key, *_ = node.keys
        if self._path_key([*path, key]) in values:
            return
        argument: ir.Value = values[self._path_key(path)]
        i64 = ir.IntegerType.get_signless(64, argument.context)
        ffi_int = ir.Type.parse("!tvm_ffi.int", context=argument.context)
        results: list[ir.Value] = [
            tvm_ffi_d.array_get_item(
                child.type,
                argument,
                tvm_ffi_d.constant(ffi_int, ir.IntegerAttr.get(i64, index)),
                element_type=child.type,
            )
            for index, child in enumerate(node.children)
        ]
        for key, result in zip(node.keys, results):
            values[self._path_key([*path, key])] = result

    def __getitem__(self, path: Sequence[int | str]) -> ir.Value | None:
        path = list(path)
        values: dict[str, ir.Value] = {
            self._path_key([name]): argument
            for name, argument in self._arguments.items()
        }
        node: _InputTreeNode = self._root
        prefix: list[int | str] = []
        for step in path:
            assert node.leaf_index is None, "path indexes through a leaf"
            if not prefix and step not in node.child_indices:
                return None
            assert step in node.child_indices, (
                f"path step {step!r} is not present in the input tree"
            )
            key, *_ = node.keys
            if prefix and self._path_key([*prefix, key]) not in values:
                self._unpack(node, prefix, values)
            child_index: int = node.child_indices[step]
            prefix.append(step)
            node = node.children[child_index]
        if self._path_key(prefix) not in values:
            self._unpack(node, prefix, values)
        return values[self._path_key(prefix)]

    def flatten(self) -> list[tuple[int, ir.Value]]:
        """Return all leaf values in the exported graph's flat order."""

        def visit(
            node: _InputTreeNode,
            path: list[int | str],
        ) -> list[tuple[int, ir.Value]]:
            if node.leaf_index is not None:
                value: ir.Value | None = self[path]
                assert value is not None, "leaf path must be present in the input tree"
                return [(node.leaf_index, value)]
            self._unpack(node, path, self._values)
            return [
                value
                for key, child in zip(node.keys, node.children)
                for value in visit(child, [*path, key])
            ]

        return [
            value
            for key, child in zip(self._root.keys, self._root.children)
            for value in visit(child, [key])
        ]


class TridentGraphModule:
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
        self._sub_modules: list[ir.Module] = []
        self.executor: Callable[..., Any] = self.stub_compile()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for _ in range(self._max_compiles):
            result = self.executor(*args, **kwargs)
            if isinstance(result, Exception):
                # The dispatcher returned an Exception ObjectRef
                # (all specializations failed); compile a new one.
                self.compile(*args, **kwargs)
            else:
                return result

        raise RuntimeError(
            f"recompilation limit ({self._max_compiles}) exceeded without finding a matching specialization"
        )

    def __name__(self) -> str:
        return self.fn.__name__

    def compile(self, *args: Any, **kwargs: Any) -> None:
        """Build a new sub-module for *args* and rebuild the combined
        module + dispatcher.  Called automatically from ``__call__`` when a
        ``GuardMatchException`` is raised."""
        # 1. Build a new sub-module for the current arguments.
        sub_mod: ir.Module = self._build_sub_module(
            self.fn, self.ctx, len(self._sub_modules), list(args), kwargs
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

        f: Callable[..., Any] = (
            tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper_from_signature(
                fn, inspect.signature(self.fn)
            )
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
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> ir.Module:
        """Export -> import -> wrap a single sub-module for *args*.

        Each sub-module's ``func.func`` is named ``main_{index}`` and its
        ``tvm_ffi.func`` is named ``{fn.__name__}_{index}`` to avoid symbol
        collisions in the merged module.
        """

        # Step 1: Export  ---------------------------------------------------
        exported_gm: torch.fx.GraphModule
        gs: torch._guards.GuardsSet
        exported_gm, gs = torch._dynamo.export(
            fn, aten_graph=True, assume_static_by_default=True
        )(*args, **kwargs)
        call_signature: inspect.Signature = inspect.signature(exported_gm.forward)
        call_bound: inspect.BoundArguments = call_signature.bind(*args, **kwargs)
        call_bound.apply_defaults()
        call_args: list[Any] = [*call_bound.arguments.values()]
        trace_gm, trace_args = copy.deepcopy((exported_gm, call_args))
        functional_gm: torch.fx.GraphModule = make_fx(
            torch.func.functionalize(trace_gm)
        )(*trace_args)
        for exported_node, functional_node in zip(
            filter(lambda node: node.op == "placeholder", exported_gm.graph.nodes),
            filter(lambda node: node.op == "placeholder", functional_gm.graph.nodes),
            strict=True,
        ):
            functional_node.meta.update(exported_node.meta)

        # Step 2: Import FX -> MLIR  ----------------------------------------
        with apply_patch():
            importer: FxImporter = FxImporter(context=ctx)
            main_func_name: Final[str] = f"main_{index}"
            main_func: func.FuncOp = importer.import_stateless_graph(
                functional_gm.graph, func_name=main_func_name
            )
            module: ir.Module = importer.module

        torch._dynamo.reset()

        # Step 3: Wrap with tvm_ffi.func.
        signature: inspect.Signature = inspect.signature(fn)
        bound: inspect.BoundArguments = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        signature_names: list[str] = [*signature.parameters]
        flat_types: Sequence[ir.Type] = main_func.type.inputs
        flat_value_is_dtype: list[bool] = [
            isinstance(value, torch.dtype)
            for value in pytree.tree_leaves((args, kwargs))
        ]

        with ctx:
            dtype_type = ir.Type.parse("!torchext.dtype", context=ctx)

        in_spec: pytree.TreeSpec | None = getattr(exported_gm, "_in_spec", None)
        assert isinstance(in_spec, pytree.TreeSpec), (
            "trident.jit requires a pytree TreeSpec for exported inputs; "
            f"got {in_spec!r}"
        )

        root_children: list[pytree.TreeSpec] = in_spec.children()
        assert in_spec.type is tuple and len(root_children) == 2, (
            f"unexpected _in_spec root (expected tuple TreeSpec, got {in_spec})"
        )
        args_spec: pytree.TreeSpec
        kwargs_spec: pytree.TreeSpec
        args_spec, kwargs_spec = root_children
        assert isinstance(args_spec, pytree.TreeSpec), (
            f"expected args tree spec, got {type(args_spec)!r}"
        )
        assert isinstance(kwargs_spec, pytree.TreeSpec), (
            f"expected kwargs tree spec, got {type(kwargs_spec)!r}"
        )

        # The ABI receives arguments in signature order.
        pos_params: list[str] = [
            name
            for name, p in signature.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        args_names: list[str] = [
            name for name, _ in zip(pos_params, args_spec.children())
        ]
        kwargs_names: list[str] = [*kwargs_spec.context]

        leaf_iter: Iterator[tuple[int, ir.Type]] = iter(enumerate(flat_types))

        def build_tree(node: pytree.TreeSpec, name: str) -> _InputTreeNode:
            """Build static pytree metadata and assign flat leaf indices."""
            if node.is_leaf():
                index, ty = next(leaf_iter)
                return _InputTreeNode(
                    dtype_type
                    if index < len(flat_value_is_dtype) and flat_value_is_dtype[index]
                    else ty,
                    leaf_index=index,
                )
            else:
                assert node.type is not dict, (
                    f"dict parameters (path step {name!r}) are not yet "
                    "supported by trident.jit"
                )
                children: list[_InputTreeNode] = [
                    build_tree(child, name) for child in node.children()
                ]
                assert not children or not all(
                    child.type == dtype_type for child in children
                ), "containers of multiple torch.dtype values are not supported"
                return _InputTreeNode(
                    ir.Type.parse("!tvm_ffi.array", context=ctx), children=children
                )

        provided_entries: dict[str, _InputTreeNode] = {
            name: build_tree(child, name)
            for name, child in [
                *zip(args_names, args_spec.children()),
                *zip(kwargs_names, kwargs_spec.children()),
            ]
        }

        entries: list[_InputTreeNode] = [
            provided_entries[name]
            if name in provided_entries
            else _InputTreeNode(
                ir.Type.parse("!tvm_ffi.array", context=ctx)
                if isinstance(bound.arguments[name], (list, tuple))
                else (
                    dtype_type
                    if isinstance(bound.arguments[name], torch.dtype)
                    else importer._cc.value_info_to_type(bound.arguments[name])
                )
            )
            for name in signature_names
        ]
        param_types: list[ir.Type] = [node.type for node in entries]
        input_root = _InputTreeNode(
            ir.Type.parse("!tvm_ffi.array", context=ctx),
            children=entries,
            keys=signature_names,
        )

        with ctx:
            # Guarded wrappers have one TVM FFI ABI result.  A main function
            # may still have multiple semantic results; those are packed into
            # a TVM FFI array in the guarded success branch below.
            normal_result_type: ir.Type
            if len(main_func.type.results) == 1:
                [result] = main_func.type.results
                normal_result_type = _convert_torch_type_to_tvm_ffi_type(result)
            else:
                normal_result_type = ir.Type.parse("!tvm_ffi.array", context=ctx)
            wrapper_result_type = ir.Type.parse(
                f"!tvm_ffi.union<{normal_result_type}, !tvm_ffi.exception>",
                context=ctx,
            )
            ffi_type: ir.FunctionType = ir.FunctionType.get(
                param_types, [wrapper_result_type]
            )

        tvm_ffi_name: Final[str] = f"{fn.__name__}_{index}"
        with ir.InsertionPoint(module.body), main_func.operation.location:
            ffi_func: tvm_ffi_d.FuncOp = tvm_ffi_d.func(
                tvm_ffi_name,
                ir.TypeAttr.get(ffi_type),
                emit_tvm_ffi_abi=ir.UnitAttr.get(ctx),
            )
            entry_block: ir.Block = ir.Block.create_at_start(ffi_func.body, param_types)
            with ir.InsertionPoint(entry_block):
                arguments = dict(zip(signature_names, entry_block.arguments))
                guard_tree = _InputTreeMap(input_root, arguments)
                guard_result: ir.Value = parse_guards(gs).build(guard_tree, ctx)
                # main_* is an ordinary func.func. Only the surrounding
                # tvm_ffi.func is exposed through the TVM FFI ABI.
                # Materialize guard success and guard failure as one ABI
                # value.  This keeps both scf.yield operations type-identical
                # while preserving the dispatcher convention that a
                # GuardMatch exception is returned from the wrapper.
                guard_if: scf.IfOp = scf.IfOp(
                    guard_result, [wrapper_result_type], has_else=True
                )
                then_block: ir.Block = guard_if.then_block
                with ir.InsertionPoint(then_block):
                    main_tree = _InputTreeMap(input_root, arguments)
                    main_args_by_index: dict[int, ir.Value] = dict(main_tree.flatten())
                    assert len(main_args_by_index) == len(flat_types), (
                        "unexpected number of reconstructed graph inputs: "
                        f"got {len(main_args_by_index)}, expected {len(flat_types)}"
                    )
                    main_args: list[ir.Value] = [
                        torchext.convert(flat_type, main_arg)
                        if main_arg.type == dtype_type
                        else main_arg
                        for flat_type, main_arg in zip(
                            flat_types, main_args_by_index.values(), strict=True
                        )
                    ]
                    call_result: ir.Value | Sequence[ir.Value] = func.call(
                        main_func.type.results,
                        main_func_name,
                        main_args,
                    )
                    result: ir.Value
                    if isinstance(call_result, ir.Value):
                        result = call_result
                    else:
                        element_types = ", ".join(
                            f"{element.type}".removeprefix("!torch.")
                            for element in call_result
                        )
                        tuple_type = ir.Type.parse(
                            f"!torch.tuple<{element_types}>",
                            context=ctx,
                        )
                        result = torch_d.prim_TupleConstruct(
                            tuple_type,
                            call_result,
                        )
                    normal_value = tvm_ffi_d.cast(wrapper_result_type, result)
                    scf.yield_([normal_value])

                else_block: ir.Block | None = guard_if.else_block
                assert else_block is not None
                with ir.InsertionPoint(else_block):
                    exception_type = ir.Type.parse("!tvm_ffi.exception", context=ctx)
                    exception = tvm_ffi_d.exception(exception_type, "GuardMatch")
                    error_value = tvm_ffi_d.cast(wrapper_result_type, exception)
                    scf.yield_([error_value])

                with ir.InsertionPoint.after(guard_if.operation):
                    tvm_ffi_d.return_(guard_if.results)

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
            needed_funcs: list[tuple[str, str]] = [
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
                handles: dict[str, ir.Value] = {}
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
                    name_ptr: ir.Value = llvm.mlir_addressof(
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
                        arg=llvm.mlir_addressof(
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
                # The lowering pass emits a packed ABI wrapper for each
                # specialization under this derived symbol.
                sub_symbol: Final[str] = f"__tvm_ffi_{self.fn.__name__}_{i}"

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
