# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import operator
from collections.abc import Iterable
from functools import partial
from typing import Any, Final, TypeAlias, cast

import torch
import triton

from trident.core import ir
from trident.core.dialects import (
    arith,
    gpu,
    torch_c,
    torchext,
)
from trident.core.dialects import (
    torch as torch_d,
)
from trident.core.extras import fx_importer
from trident.core.extras.fx_importer import ContextCache, FxImporter, GraphNodeImporter

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


def _resolve_triton_binder_value(
    name: str,
    value: KernelArgument,
) -> KernelValue:
    if isinstance(value, torch.fx.Node):
        assert "val" in value.meta, (
            f"Triton binder argument {name} node {value.name} "
            "does not contain metadata value"
        )
        metadata_value = value.meta["val"]
        if isinstance(metadata_value, torch.Tensor):
            return triton.MockTensor(dtype=metadata_value.dtype)
        return _resolve_triton_binder_value(name, metadata_value)

    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return _resolve_triton_binder_value(name, cast(KernelValue, value.node.hint))

    if isinstance(value, tuple):
        items = tuple(_resolve_triton_binder_value(name, item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return items

    if isinstance(value, list):
        return [_resolve_triton_binder_value(name, item) for item in value]

    if isinstance(value, dict):
        return {
            key: _resolve_triton_binder_value(name, item) for key, item in value.items()
        }

    return value


class TridentGraphNodeImporter(GraphNodeImporter):
    def __init__(
        self,
        fx_importer: Any,
        context: ir.Context,
        context_cache: Any,
        block: ir.Block,
        specialization_id: int = 0,
    ) -> None:
        super().__init__(fx_importer, context, context_cache, block)
        self.specialization_id: int = specialization_id

    @staticmethod
    def _import_kernel_value(
        importer: GraphNodeImporter,
        loc: ir.Location,
        value: KernelArgument,
    ) -> ir.Value | None:
        import_torch_int_op = lambda name, operands: (
            torch_d.operator(
                [torch_d.TorchIntType.get(loc.context)],
                name,
                operands,
                0,
                loc=loc,
            ).result
        )

        arithmetic_ops = {
            operator.add: torch_d.aten_add_int,
            operator.floordiv: torch_d.aten_floordiv_int,
            operator.mul: torch_d.aten_mul_int,
            operator.or_: lambda lhs, rhs, *, loc: import_torch_int_op(
                "torch.aten.__or__.int", [lhs, rhs]
            ),
            operator.rshift: lambda lhs, rhs, *, loc: import_torch_int_op(
                "torch.aten.__rshift__.int", [lhs, rhs]
            ),
            operator.sub: torch_d.aten_sub_int,
            torch.sym_max: torch_d.prim_max_int,
            torch.sym_min: torch_d.prim_min_int,
        }

        if isinstance(value, torch.fx.Node):
            if value.target is torch.sym_sum:
                arguments = value.args
                if len(arguments) == 1 and all(
                    isinstance(argument, (list, tuple)) for argument in arguments
                ):
                    [arguments] = arguments
                result = torch_d.constant_int(0, loc=loc)
                for argument in arguments:
                    imported = TridentGraphNodeImporter._import_kernel_value(
                        importer, loc, argument
                    )
                    assert imported is not None
                    result = arithmetic_ops[operator.add](result, imported, loc=loc)
                return result

            if value.target in (
                torch.ops.aten.sym_size.int,
                torch.ops.aten.sym_stride.int,
            ):
                tensor, index = value.args
                tensor = importer._import_argument(loc, tensor)
                imported_index = TridentGraphNodeImporter._import_kernel_value(
                    importer, loc, index
                )
                assert imported_index is not None
                index = torch_c.to_i64(imported_index, loc=loc)
                if value.target is torch.ops.aten.sym_size.int:
                    native_value = torchext.tensor_size(tensor, index, loc=loc)
                else:
                    native_value = torchext.tensor_stride(tensor, index, loc=loc)
                return torch_c.from_i64(
                    native_value,
                    loc=loc,
                )

            if value.target in arithmetic_ops:
                lhs, rhs = (
                    TridentGraphNodeImporter._import_kernel_value(
                        importer, loc, argument
                    )
                    for argument in value.args
                )
                assert lhs is not None and rhs is not None
                return arithmetic_ops[value.target](lhs, rhs, loc=loc)

            if value.target is operator.pow:
                base, exponent = value.args
                assert isinstance(exponent, int) and exponent >= 0
                base = TridentGraphNodeImporter._import_kernel_value(
                    importer, loc, base
                )
                assert base is not None
                result = torch_d.constant_int(1, loc=loc)
                for _ in range(exponent):
                    result = arithmetic_ops[operator.mul](result, base, loc=loc)
                return result

            value = value.meta.get("val")

        if isinstance(value, torch.SymInt):
            expression = value.node.expr
            assert expression.free_symbols, (
                f"cannot lower unsupported symbolic Triton integer expression {expression!s}; refusing to use its concrete hint"
            )
            value = ast.literal_eval(value)
        if isinstance(value, int):
            return torch_d.constant_int(value, loc=loc)
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

        if TridentGraphNodeImporter._get_symbolic_integer(value) is not None:
            imported = TridentGraphNodeImporter._import_kernel_value(
                importer, loc, value
            )
            assert imported is not None, (
                f"symbolic Triton kernel argument could not be lowered: {value}"
            )
            return imported
        else:
            imported = importer._import_argument(loc, value)
            assert imported is not None
            return imported

    @staticmethod
    def _get_symbolic_integer(value: KernelArgument) -> torch.SymInt | None:
        if isinstance(value, torch.fx.Node):
            value = value.meta.get("val")
        elif isinstance(value, torch.SymInt):
            return value
        else:
            return None

    def _import_symbolic_torch_op(
        self,
        loc: ir.Location,
        node: torch.fx.Node,
        target: Any,
    ) -> None:
        if target is operator.pow and isinstance(node.meta.get("val"), torch.SymInt):
            base, exponent = node.args
            assert isinstance(exponent, int) and exponent >= 0
            base = self._import_argument(loc, base)
            result = torch_d.constant_int(1, loc=loc)
            for _ in range(exponent):
                result = torch_d.aten_mul_int(result, base, loc=loc)
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
                result = torch_d.aten_add_int(result, operand, loc=loc)
            self.bind_node_value(node, result)
        elif target in (torch.sym_max, torch.sym_min):
            lhs, rhs = (self._import_argument(loc, argument) for argument in node.args)
            operation = (
                torch_d.prim_max_int
                if target is torch.sym_max
                else torch_d.prim_min_int
            )
            self.bind_node_value(node, operation(lhs, rhs, loc=loc))
        elif target in (operator.or_, operator.rshift):
            lhs, rhs = (self._import_argument(loc, argument) for argument in node.args)
            op_name = (
                "torch.aten.__or__.int"
                if target is operator.or_
                else "torch.aten.__rshift__.int"
            )
            operation = torch_d.operator(
                [torch_d.TorchIntType.get(loc.context)], op_name, [lhs, rhs], 0, loc=loc
            )
            self.bind_node_value(node, operation.result)
        else:
            super()._import_symbolic_torch_op(loc, node, target)

    def _import_hop_triton_kernel_wrapper_functional(
        self, loc: ir.Location, node: torch.fx.Node, hop: Any
    ) -> None:
        _import_hop_triton_kernel_wrapper(self, loc, node, hop)

    def _import_hop_triton_kernel_wrapper_mutation(
        self, loc: ir.Location, node: torch.fx.Node, hop: Any
    ) -> None:
        _import_hop_triton_kernel_wrapper(self, loc, node, hop)

    def import_nodes(
        self,
        nodes: Iterable[torch.fx.Node],
        *,
        skip_placeholders_outputs: bool = False,
        import_symbolic_shape_expressions: bool = False,
    ) -> None:
        with ir.InsertionPoint(self._b):
            loc = ir.Location.unknown()
            if import_symbolic_shape_expressions:
                symbolic_guards = self._cc.get_symbolic_guards()
                self._import_shape_symbols_with_guards(loc, symbolic_guards)

            num_placeholders = 0
            for node in nodes:
                node_loc = self._cc.get_node_location(node)
                if node_loc is not None:
                    loc = node_loc
                if node.op == "placeholder" and not skip_placeholders_outputs:
                    self.bind_node_value(node, self._b.arguments[num_placeholders])
                    num_placeholders += 1
                elif node.op == "call_function":
                    target = node.target
                    if target == operator.getitem:
                        self._import_getitem(loc, node)
                    elif (
                        target in fx_importer.SYMBOLIC_TORCH_OPS
                        or target
                        in (
                            torch.sym_max,
                            torch.sym_min,
                            torch.sym_sum,
                        )
                        or (
                            fx_importer.is_symbolic(node.meta.get("val"))
                            and fx_importer.is_builtin_function_or_method(target)
                        )
                    ):
                        self._import_symbolic_torch_op(loc, node, target)
                    elif isinstance(target, torch._ops.OpOverload):
                        self._import_torch_op_overload(loc, node)
                    elif isinstance(target, torch._ops.HigherOrderOperator):
                        self._import_hop(loc, node, target)
                    else:
                        raise NotImplementedError(
                            f"FIX ME: Unimplemented call_function: target={target}, {node.meta}"
                        )
                elif node.op == "output" and not skip_placeholders_outputs:
                    output_arg = node.args[0]
                    result_nodes = (
                        output_arg
                        if isinstance(output_arg, (list, tuple))
                        else [output_arg]
                    )
                    operands = [self._import_argument(loc, arg) for arg in result_nodes]
                    ir.Operation.create("func.return", operands=operands, loc=loc)
                if import_symbolic_shape_expressions:
                    self._create_bind_symbolic_shape_ops(loc, node)


def _import_hop_triton_kernel_wrapper(
    self: GraphNodeImporter,
    loc: ir.Location,
    node: torch.fx.Node,
    hop: Any,
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
        parameter.name: _resolve_triton_binder_value(
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
    operands: dict[str, tuple[ir.Attribute, ir.Value]] = {}

    def import_constant(value: KernelValue, name: str) -> ir.Value | None:
        with loc:
            if isinstance(value, tuple):
                elements = [
                    import_constant(element, f"{name}[{index}]")
                    for index, element in enumerate(value)
                ]
                element_types = ", ".join(
                    f"{element.type}".removeprefix("!torch.") for element in elements
                )
                tuple_type = ir.Type.parse(
                    f"!torch.tuple<{element_types}>", context=loc.context
                )
                return torch_d.prim_TupleConstruct(tuple_type, elements, loc=loc)
            if isinstance(value, bool):
                return torch_d.constant_bool(value)
            if isinstance(value, int):
                return torch_d.constant_int(value)
            if isinstance(value, float):
                return torch_d.constant_float(value)
            if isinstance(value, str):
                return torch_d.constant_str(value)

    def constant_value_attribute(value: Any, name: str) -> ir.Attribute | None:
        if isinstance(value, tuple):
            return ir.ArrayAttr.get(
                [
                    constant_value_attribute(element, f"{name}[{index}]")
                    for index, element in enumerate(value)
                ]
            )
        if isinstance(value, bool):
            return ir.BoolAttr.get(value)
        if isinstance(value, int):
            assert -(1 << 63) <= value < (1 << 63), (
                f"constexpr specialization {name!r} does not fit in i64: {value}"
            )
            return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
        if isinstance(value, float):
            return ir.FloatAttr.get_f64(value)
        if isinstance(value, str):
            return ir.StringAttr.get(value)

    def constant_specialization(value: Any, name: str) -> ir.Attribute:
        value_attr = constant_value_attribute(value, name)
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
            f"compiled Triton signature does not match its source parameter order: expected {(name, triton_type)!r}, got {(compiled_name, compiled_type)!r}"
        )

        if triton_type == "constexpr":
            if name in knodes:
                operand = TridentGraphNodeImporter._import_kernel_argument(
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
            elif triton_type in {
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
            }:
                native_type = f"i{triton_type[1:]}"
            elif triton_type in {"fp32", "fp64"}:
                native_type = f"f{triton_type[2:]}"
            else:
                raise RuntimeError(f"unsupported Triton argument type: {triton_type}")

            if name in knodes:
                operand = TridentGraphNodeImporter._import_kernel_argument(
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

        assert operand is not None
        operands[name] = (specialization_attr, operand)
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
    binary_name: Final[str] = f"_trident_s{self.specialization_id}_{node.name}"
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
    i32_type = ir.IntegerType.get_signless(32)
    grid_x, grid_y, grid_z = grid

    def import_launch_value(value: KernelArgument) -> ir.Value:
        imported = TridentGraphNodeImporter._import_kernel_value(self, loc, value)
        assert imported is not None
        return torch_c.to_i64(imported, loc=loc)

    def import_launch_constant(value: int) -> ir.Value:
        return torch_c.to_i64(torch_d.constant_int(value, loc=loc), loc=loc)

    torchext.trident_kernel_launch(
        ir.Attribute.parse(f"@{binary_name}::@{kernel.metadata.name}"),
        import_launch_value(grid_x),
        import_launch_value(grid_y),
        import_launch_value(grid_z),
        import_launch_constant(kernel.metadata.num_warps * kernel.metadata.warp_size),
        import_launch_constant(1),
        import_launch_constant(1),
        [operand for _, operand in operands.values()],
        ir.ArrayAttr.get([specialization for specialization, _ in operands.values()]),
        dynamic_shared_memory_size=arith.constant(
            i32_type, kernel.metadata.shared, loc=loc
        ),
        loc=loc,
    )
    self._multi_result_nodes.add(node)

    for name in output_names:
        _, operand = operands[name]
        self.bind_node_value(node, operand, name)


class TridentContextCache(ContextCache):
    def value_info_to_type(
        self,
        val: Any,
        *,
        tensor_meta: Any = None,
        mutable: bool = False,
    ) -> ir.Type:
        if isinstance(val, torch.dtype):
            return self.torch_int_type
        return super().value_info_to_type(val, tensor_meta=tensor_meta, mutable=mutable)


class TridentFxImporter(FxImporter):
    def __init__(self, *, context: ir.Context, specialization_id: int):
        super().__init__(
            context=context,
            graph_node_importer_cls=cast(
                Any,
                partial(TridentGraphNodeImporter, specialization_id=specialization_id),
            ),
        )
        self._cc = TridentContextCache(self._c, py_attr_tracker=self._py_attr_tracker)

    def _graph_to_function_meta(
        self, graph: torch.fx.Graph
    ) -> tuple[ir.FunctionType, ir.Location]:
        input_types: list[ir.Type] = []
        result_types: list[ir.Type] = []
        loc: ir.Location | None = None
        for node in graph.nodes:
            if loc is None:
                loc = self._cc.get_node_location(node)
            if node.op == "placeholder":
                input_types.append(self._cc.node_val_to_type(node))
            elif node.op == "output":
                output_arg = node.args[0]
                result_nodes = (
                    output_arg
                    if isinstance(output_arg, (list, tuple))
                    else [output_arg]
                )
                for result_node in result_nodes:
                    if result_node is None:
                        result_types.append(ir.Type.parse("!torch.none", self._c))
                    elif isinstance(result_node, torch.Tensor):
                        result_types.append(
                            self._cc.tensor_to_vtensor_type(result_node)
                        )
                    elif isinstance(result_node, torch.fx.Node):
                        result_types.append(self._cc.node_val_to_type(result_node))
                    else:
                        result_types.append(self._cc.value_info_to_type(result_node))
        return (
            ir.FunctionType.get(input_types, result_types, context=self._c),
            loc or ir.Location.unknown(self._c),
        )
