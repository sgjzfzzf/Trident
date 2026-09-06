# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import operator
import threading
from collections.abc import Callable, Mapping, MutableMapping, MutableSet
from collections.abc import Set as AbstractSet
from contextlib import ExitStack
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Final, Self, TypeAlias, cast

import numpy as np
import torch
import triton

from trident.core import ir
from trident.core.dialects import (
    arith,
    gpu,
    torchext,
)
from trident.core.dialects import (
    torch as torch_d,
)
from trident.core.extras import fx_importer
from trident.core.extras.fx_importer import GraphNodeImporter

KernelScalar: TypeAlias = None | bool | int | float | str | torch.dtype | torch.device
KernelValue: TypeAlias = (
    torch.Tensor
    | torch.SymInt
    | torch.SymFloat
    | torch.SymBool
    | KernelScalar
    | tuple["KernelValue", ...]
    | list["KernelValue"]
    | dict[str, "KernelValue"]
)
KernelArgument: TypeAlias = torch.fx.Node | KernelValue

_MISSING: Final[object] = object()


class GraphNodeImporterTritonHopPatchState:
    def __init__(self, specialization_id: int = 0) -> None:
        self.specialization_id: int = specialization_id
        self._token: Token[GraphNodeImporterTritonHopPatchState | None] | None = None

    @staticmethod
    def _resolve_triton_binder_value(
        name: str,
        value: KernelArgument,
        path: str = "",
    ) -> KernelValue:
        location = f"{name}{path}"
        if isinstance(value, torch.fx.Node):
            assert "val" in value.meta, (
                f"Triton binder argument {location} node {value.name} "
                "does not contain metadata value"
            )
            metadata_value = value.meta["val"]
            if isinstance(metadata_value, torch.Tensor):
                return triton.MockTensor(dtype=metadata_value.dtype)
            return GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                location, metadata_value
            )

        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                location, cast(KernelValue, value.node.hint)
            )

        if isinstance(value, tuple):
            items = tuple(
                GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                    name, item, f"{path}[{index}]"
                )
                for index, item in enumerate(value)
            )
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return items

        if isinstance(value, list):
            return [
                GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                    name, item, f"{path}[{index}]"
                )
                for index, item in enumerate(value)
            ]

        if isinstance(value, dict):
            return {
                key: GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                    name, item, f"{path}[{key!r}]"
                )
                for key, item in value.items()
            }

        return value

    @staticmethod
    def _import_kernel_value(
        importer: GraphNodeImporter,
        loc: ir.Location,
        value: KernelArgument,
        *,
        torch_scalar: bool = False,
    ) -> ir.Value | None:
        i64_type = ir.IntegerType.get_signless(64)
        if torch_scalar:
            arithmetic_ops = {
                operator.add: torch_d.aten_add_int,
                operator.floordiv: torch_d.aten_floordiv_int,
                operator.mul: torch_d.aten_mul_int,
                operator.sub: torch_d.aten_sub_int,
                torch.sym_max: torch_d.prim_max_int,
                torch.sym_min: torch_d.prim_min_int,
            }
            shape_ops = {torch.ops.aten.sym_size.int: torch_d.aten_size_int}
        else:
            arithmetic_ops = {
                operator.add: arith.addi,
                operator.floordiv: arith.floordivsi,
                operator.mul: arith.muli,
                operator.or_: arith.ori,
                operator.rshift: arith.shrsi,
                operator.sub: arith.subi,
                torch.sym_max: arith.maxsi,
                torch.sym_min: arith.minsi,
            }
            shape_ops = {
                torch.ops.aten.sym_size.int: torchext.tensor_size,
                torch.ops.aten.sym_stride.int: torchext.tensor_stride,
            }

        def import_constant(constant: int) -> ir.Value:
            if torch_scalar:
                return torch_d.constant_int(constant, loc=loc)
            return arith.constant(i64_type, constant, loc=loc)

        def import_value(argument: KernelArgument) -> ir.Value:
            result = GraphNodeImporterTritonHopPatchState._import_kernel_value(
                importer, loc, argument, torch_scalar=torch_scalar
            )
            assert result is not None
            return result

        if isinstance(value, torch.fx.Node):
            if value.target is torch.sym_sum:
                arguments = value.args
                if len(arguments) == 1 and all(
                    isinstance(argument, (list, tuple)) for argument in arguments
                ):
                    [arguments] = arguments
                result = import_constant(0)
                for argument in arguments:
                    result = arithmetic_ops[operator.add](
                        result, import_value(argument), loc=loc
                    )
                return result

            if value.target in shape_ops:
                tensor, index = value.args
                tensor = importer._import_argument(loc, tensor)
                return shape_ops[value.target](tensor, import_constant(index), loc=loc)

            if value.target in arithmetic_ops:
                lhs, rhs = (import_value(argument) for argument in value.args)
                return arithmetic_ops[value.target](lhs, rhs, loc=loc)

            if value.target is operator.pow:
                base, exponent = value.args
                assert isinstance(exponent, int) and exponent >= 0
                base = import_value(base)
                result = import_constant(1)
                for _ in range(exponent):
                    result = arithmetic_ops[operator.mul](result, base, loc=loc)
                return result

            value = value.meta.get("val")

        if isinstance(value, torch.SymInt):
            expression = value.node.expr
            if expression.free_symbols:
                value_kind = "kernel argument" if torch_scalar else "launch expression"
                raise ValueError(
                    f"cannot lower unsupported symbolic Triton {value_kind} "
                    f"{expression!s}; refusing to use its concrete hint"
                )
            value = ast.literal_eval(value)
        if isinstance(value, int):
            return import_constant(value)
        return None

    @staticmethod
    def _import_kernel_argument(
        importer: GraphNodeImporter,
        loc: ir.Location,
        value: KernelArgument,
    ) -> ir.Value:
        """Import a Triton kernel argument without concretizing SymInts.

        The generic FX importer is correct for ordinary graph values, but a
        Triton HOP also carries scalar launch parameters such as ``m`` and
        ``N``.  Those values can be symbolic arithmetic nodes whose concrete
        example value is not a valid runtime replacement.  Route symbolic
        integer expressions through the same lowering used for launch grids.
        """

        if (
            GraphNodeImporterTritonHopPatchState._get_symbolic_integer(value)
            is not None
        ):
            imported = GraphNodeImporterTritonHopPatchState._import_kernel_value(
                importer, loc, value, torch_scalar=True
            )
            assert imported is not None, (
                f"symbolic Triton kernel argument could not be lowered: {value}"
            )
            return imported
        imported = importer._import_argument(loc, value)
        assert imported is not None
        return imported

    @staticmethod
    def _get_symbolic_integer(value: KernelArgument) -> torch.SymInt | None:
        if isinstance(value, torch.fx.Node):
            value = value.meta.get("val")
        return value if isinstance(value, torch.SymInt) else None

    def __enter__(self) -> Self:
        _patch_manager.apply(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _patch_manager.restore(self)


class _GraphNodeImporterPatchManager:
    """Own the process-global monkey-patch lifecycle."""

    def __init__(self) -> None:
        self.refcount: int = 0
        self.patches: ExitStack | None = None
        self.original_import_symbolic_torch_op: Callable[..., None] | None = None
        self.original_return_node_values: Callable[..., None] | None = None
        self._active_state: ContextVar[GraphNodeImporterTritonHopPatchState | None] = (
            ContextVar("trident_importer_patch_state", default=None)
        )
        self.lock: Final[threading.RLock] = threading.RLock()

    @property
    def active_state(self) -> GraphNodeImporterTritonHopPatchState | None:
        return self._active_state.get()

    @staticmethod
    def _patch_attribute(
        patches: ExitStack,
        target: object,
        value: Callable[..., object],
    ) -> object:
        """Replace one attribute and register its exact restoration."""
        attribute = value.__name__
        original = getattr(target, attribute, _MISSING)
        setattr(target, attribute, value)
        if original is _MISSING:
            patches.callback(delattr, target, attribute)
        else:
            patches.callback(setattr, target, attribute, original)
        return original

    @staticmethod
    def _patch_mapping(
        patches: ExitStack,
        mapping: MutableMapping[object, object],
        additions: Mapping[object, object],
    ) -> None:
        conflicts = mapping.keys() & additions.keys()
        assert not conflicts, f"cannot patch existing mapping keys: {conflicts}"
        original = mapping.copy()
        patches.callback(mapping.update, original)
        patches.callback(mapping.clear)
        mapping.update(additions)

    @staticmethod
    def _patch_set(
        patches: ExitStack,
        values: MutableSet[object],
        additions: AbstractSet[object],
    ) -> None:
        conflicts = values & additions
        assert not conflicts, f"cannot patch existing set values: {conflicts}"
        original = values.copy()
        patches.callback(values.update, original)
        patches.callback(values.clear)
        values.update(additions)

    def _install_patches(
        self,
        patches: ExitStack,
    ) -> tuple[Callable[..., None], Callable[..., None]]:
        for patch in (
            _import_hop_triton_kernel_wrapper_functional,
            _import_hop_triton_kernel_wrapper_mutation,
        ):
            self._patch_attribute(patches, GraphNodeImporter, patch)
        original_import_symbolic_torch_op = self._patch_attribute(
            patches, GraphNodeImporter, _import_symbolic_torch_op
        )
        original_return_node_values = self._patch_attribute(
            patches, GraphNodeImporter, return_node_values
        )
        assert callable(original_import_symbolic_torch_op)
        assert callable(original_return_node_values)

        self._patch_mapping(
            patches,
            fx_importer.SCALAR_TYPE_TO_TORCH_MLIR_TYPE,
            {torch.dtype: "!torch.int"},
        )
        self._patch_mapping(
            patches,
            fx_importer.TORCH_DTYPE_TO_INT,
            {torch.uint32: 28},
        )
        self._patch_mapping(
            patches,
            fx_importer.TORCH_DTYPE_TO_MLIR_TYPE,
            {torch.uint32: lambda: ir.IntegerType.get_unsigned(32)},
        )
        self._patch_mapping(
            patches,
            fx_importer.TORCH_DTYPE_TO_MLIR_TYPE_ASM,
            {torch.uint32: "ui32"},
        )
        self._patch_mapping(
            patches,
            fx_importer.TORCH_DTYPE_TO_NPY_TYPE,
            {torch.uint32: np.uint32},
        )
        self._patch_mapping(
            patches,
            fx_importer.PY_BUILTIN_TO_TORCH_OP,
            {
                operator.or_: torch.ops.aten.__or__,
                operator.pow: torch.ops.aten.pow,
                operator.rshift: torch.ops.aten.__rshift__,
            },
        )
        self._patch_set(
            patches,
            fx_importer.SYMBOLIC_TORCH_OPS,
            {torch.sym_max, torch.sym_min, torch.sym_sum},
        )
        return original_import_symbolic_torch_op, original_return_node_values

    def apply(self, state: GraphNodeImporterTritonHopPatchState) -> None:
        with self.lock:
            assert state._token is None, "patch state is already active"
            if self.refcount == 0:
                patches = ExitStack()
                try:
                    (
                        self.original_import_symbolic_torch_op,
                        self.original_return_node_values,
                    ) = self._install_patches(patches)
                except BaseException:
                    patches.close()
                    raise
                self.patches = patches
            self.refcount += 1
            state._token = self._active_state.set(state)

    def restore(self, state: GraphNodeImporterTritonHopPatchState) -> None:
        with self.lock:
            token = state._token
            assert token is not None, "patch state is not active"
            assert self.active_state is state, "patch contexts must exit in LIFO order"
            self._active_state.reset(token)
            state._token = None
            assert self.refcount > 0
            self.refcount -= 1
            if self.refcount > 0:
                return
            patches = self.patches
            assert patches is not None
            patches.close()
            self.patches = None
            self.original_import_symbolic_torch_op = None
            self.original_return_node_values = None


_patch_manager = _GraphNodeImporterPatchManager()


def _import_symbolic_torch_op(
    self: GraphNodeImporter,
    loc: ir.Location,
    node: torch.fx.Node,
    target: object,
) -> None:
    if target is operator.pow and isinstance(node.meta.get("val"), torch.SymInt):
        base, exponent = node.args
        assert isinstance(exponent, int) and exponent >= 0
        base = self._import_argument(loc, base)
        result = torch_d.constant_int(1, loc=loc)
        for _ in range(exponent):
            result = torch_d.aten_mul_int(
                result,
                base,
                loc=loc,
            )
        self.bind_node_value(node, result)
    elif target is torch.sym_sum:
        arguments = node.args
        if len(arguments) == 1 and all(
            isinstance(argument, (list, tuple)) for argument in arguments
        ):
            [arguments] = arguments
        operands = [self._import_argument(loc, argument) for argument in arguments]
        result = torch_d.constant_int(0, loc=loc)
        for operand in operands:
            result = torch_d.aten_add_int(
                result,
                operand,
                loc=loc,
            )
        self.bind_node_value(node, result)
    elif target in (torch.sym_max, torch.sym_min):
        lhs, rhs = (self._import_argument(loc, argument) for argument in node.args)
        operation = (
            torch_d.prim_max_int if target is torch.sym_max else torch_d.prim_min_int
        )
        result = operation(lhs, rhs, loc=loc)
        self.bind_node_value(node, result)
    else:
        original = _patch_manager.original_import_symbolic_torch_op
        assert original is not None
        original(self, loc, node, target)


def _import_hop_triton_kernel_wrapper(
    self: GraphNodeImporter,
    loc: ir.Location,
    node: torch.fx.Node,
    hop: object,
) -> None:
    knodes = cast(dict[str, KernelArgument], node.kwargs["kwargs"])
    output_names: list[str] = node.kwargs.get("tensors_to_clone", [])
    constant_args_idx: Final[int] = node.kwargs["constant_args_idx"]
    constant_args = cast(
        dict[str, KernelValue],
        torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_constant_args(
            constant_args_idx
        ),
    )
    kernel_idx: Final[int] = node.kwargs["kernel_idx"]
    function: triton.KernelInterface = (
        torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
            kernel_idx
        )
    )
    device = triton.runtime.driver.active.get_current_device()
    configs: list[triton.Config] = getattr(function, "configs", [])
    best_config: triton.Config | None = getattr(function, "best_config", None)
    while not isinstance(function, triton.JITFunction):
        function = function.fn
    kernel_cache, kernel_key_cache, _, _, binder = function.device_caches[device]
    values = {**knodes, **constant_args}
    binder_args = {
        parameter.name: GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
            parameter.name, values[parameter.name]
        )
        for parameter in function.params
        if parameter.name in values
    }
    config_kwargs = {} if best_config is None else best_config.all_kwargs()
    runtime_kwargs = {
        "debug": binder_args.get("debug", function.debug) or triton.knobs.runtime.debug,
        "instrumentation_mode": triton.knobs.compilation.instrumentation_mode,
    }
    bound_args, specialization, options = binder(
        **binder_args, **config_kwargs, **runtime_kwargs
    )
    key: str = triton.runtime.jit.compute_cache_key(
        kernel_key_cache, specialization, options
    )
    kernel: triton.compiler.CompiledKernel | None = kernel_cache.get(key)
    assert kernel is not None, f"failed to get compiled Triton kernel for {node.name}"
    integer_types = {
        "i1",
        "u1",
        "i8",
        "u8",
        "i16",
        "u16",
        "i32",
        "u32",
        "i64",
        "u64",
    }
    arg_attrs: list[ir.DictAttr] = []
    operands: list[ir.Value] = []
    operands_by_name: dict[str, ir.Value] = {}

    def import_constant(value: KernelValue, name: str) -> ir.Value:
        # TODO: Support string constexpr specializations, such as the
        # ACTIVATION argument in examples/mm.py.
        assert type(value) in {bool, int, float}, (
            f"unsupported constexpr argument {name!r} of type "
            f"{type(value).__name__}; expected bool, int, or float"
        )
        with loc:
            if type(value) is bool:
                return torch_d.constant_bool(value)
            if type(value) is int:
                return torch_d.constant_int(value)
            return torch_d.constant_float(value)

    def constant_specialization(value: object, name: str) -> ir.Attribute:
        assert type(value) in {bool, int, float}, (
            f"unsupported constexpr specialization {name!r} of type "
            f"{type(value).__name__}; expected bool, int, or float"
        )
        if type(value) is bool:
            value_attr = ir.BoolAttr.get(value)
        elif type(value) is int:
            assert -(1 << 63) <= value < (1 << 63), (
                f"constexpr specialization {name!r} does not fit in i64: {value}"
            )
            value_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
        else:
            value_attr = ir.FloatAttr.get_f64(value)
        return ir.Attribute.parse(
            f"#torchext.constant_specialization<value = {value_attr}>"
        )

    for parameter, (triton_type, specialization_descriptor), compiled_parameter in zip(
        function.params,
        specialization,
        kernel.src.signature.items(),
        strict=True,
    ):
        name = parameter.name
        compiled_name, compiled_type = compiled_parameter
        assert (compiled_name, compiled_type) == (name, triton_type), (
            "compiled Triton signature does not match its source parameter order: "
            f"expected {(name, triton_type)!r}, got "
            f"{(compiled_name, compiled_type)!r}"
        )

        if triton_type == "constexpr":
            if name in knodes:
                operand = GraphNodeImporterTritonHopPatchState._import_kernel_argument(
                    self, loc, knodes[name]
                )
            else:
                operand = import_constant(bound_args[name], name)
            specialization_attr = constant_specialization(
                specialization_descriptor, name
            )
        else:
            if triton_type.startswith("*"):
                native_type = "!llvm.ptr"
            elif triton_type in integer_types:
                native_type = f"i{ast.literal_eval(triton_type[1:])}"
            elif triton_type in {"fp32", "fp64"}:
                native_type = f"f{triton_type[2:]}"
            else:
                raise RuntimeError(f"unsupported Triton argument type: {triton_type}")

            if name in knodes:
                operand = GraphNodeImporterTritonHopPatchState._import_kernel_argument(
                    self, loc, knodes[name]
                )
            elif name in bound_args:
                operand = import_constant(bound_args[name], name)
            else:
                raise RuntimeError(
                    f"missing runtime argument for {name} of type {triton_type}"
                )

            specialization_attr = ir.Attribute.parse(
                "#torchext.variable_specialization<"
                f"kind = {native_type}"
                f"{', divisibility = 16' if specialization_descriptor == 'D' else ''}>"
            )

        operands.append(operand)
        operands_by_name[name] = operand
        arg_attrs.append(
            ir.DictAttr.get({"triton.specialization": specialization_attr})
        )
    grids: list[tuple[int, int, int]] = node.kwargs["grid"]
    if len(configs) > 0 and best_config is not None:
        i: Final[int] = configs.index(best_config)
        grid: tuple[int, int, int] = grids[i]
    else:
        [grid] = grids
    # Each imported specialization gets a unique symbol namespace from the
    # graph module.  FX node names are unique within an imported graph, so the
    # pair is sufficient to keep binaries distinct after module merging.
    cubin = kernel.asm["cubin"]
    active_state = _patch_manager.active_state
    assert active_state is not None
    binary_name: Final[str] = f"_trident_s{active_state.specialization_id}_{node.name}"
    module_op = self.fx_importer.module.operation
    module_op.attributes["gpu.container_module"] = ir.UnitAttr.get()
    if all(
        not isinstance(op, gpu.BinaryOp) or op.sym_name != binary_name
        for op in self.fx_importer.module.body.operations
    ):
        cubin_mlir: str = "".join(f"\\{byte:02X}" for byte in cubin)
        gpu_object = ir.Attribute.parse(
            f'#gpu.object<#nvvm.target<chip = "sm_{kernel.metadata.target.arch}">, '
            f'"{cubin_mlir}">'
        )
        with ir.InsertionPoint(self.fx_importer.module.body):
            gpu.binary(
                binary_name,
                ir.ArrayAttr.get([gpu_object]),
                offloading_handler=ir.Attribute.parse("#gpu.select_object"),
                loc=loc,
            )
    i64_type = ir.IntegerType.get_signless(64)
    i32_type = ir.IntegerType.get_signless(32)
    grid_x, grid_y, grid_z = grid

    launch = torchext.trident_kernel_launch(
        ir.Attribute.parse(f"@{binary_name}::@{kernel.metadata.name}"),
        GraphNodeImporterTritonHopPatchState._import_kernel_value(self, loc, grid_x),
        GraphNodeImporterTritonHopPatchState._import_kernel_value(self, loc, grid_y),
        GraphNodeImporterTritonHopPatchState._import_kernel_value(self, loc, grid_z),
        arith.constant(
            i64_type,
            kernel.metadata.num_warps * kernel.metadata.warp_size,
            loc=loc,
        ),
        arith.constant(i64_type, 1, loc=loc),
        arith.constant(i64_type, 1, loc=loc),
        operands,
        dynamic_shared_memory_size=arith.constant(
            i32_type, kernel.metadata.shared, loc=loc
        ),
        loc=loc,
    )
    launch.attributes["arg_attrs"] = ir.ArrayAttr.get(arg_attrs)

    self._multi_result_nodes.add(node)

    for name in output_names:
        self.bind_node_value(node, operands_by_name[name], name)


def _import_hop_triton_kernel_wrapper_functional(
    self: GraphNodeImporter,
    loc: ir.Location,
    node: torch.fx.Node,
    hop: object,
) -> None:
    _import_hop_triton_kernel_wrapper(self, loc, node, hop)


def _import_hop_triton_kernel_wrapper_mutation(
    self: GraphNodeImporter,
    loc: ir.Location,
    node: torch.fx.Node,
    hop: object,
) -> None:
    _import_hop_triton_kernel_wrapper(self, loc, node, hop)


def return_node_values(
    self: GraphNodeImporter,
    loc: ir.Location,
    nodes: list[torch.fx.Node | None],
    constants: dict[int, KernelValue],
) -> None:
    """Fix constant output indices before delegating to the FX importer."""
    original_return_node_values = _patch_manager.original_return_node_values
    assert original_return_node_values is not None
    compact_constants = {
        compact_index: constants[original_index]
        for compact_index, original_index in zip(
            (index for index, node in enumerate(nodes) if node is None),
            sorted(constants.keys()),
            strict=True,
        )
    }
    original_return_node_values(self, loc, nodes, compact_constants)


def apply_patch(specialization_id: int = 0) -> GraphNodeImporterTritonHopPatchState:
    return GraphNodeImporterTritonHopPatchState(specialization_id)
