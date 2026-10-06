# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Translate FX nodes directly to semantic TVM FFI and native arithmetic IR.

Unlike a value-semantic tensor importer, this importer preserves the original
ATen call order and tensor aliases. It does not functionalize mutations.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from functools import reduce
from typing import Any

import torch
from torch.export.graph_signature import ConstantArgument, InputKind, OutputKind

from trident.core import ir
from trident.core.dialects import arith, func, scf
from trident.core.dialects import tvm_ffi as ffi

from .ir_utils import (
    array,
    array_get,
    binary,
    box,
    checked_call,
    constant,
    ffi_type,
    native,
    native_constant,
    value_type,
)
from .triton_importer import import_kernel

_SCALAR_OPERATIONS: dict[Callable[..., Any], Callable[..., Any]] = {
    operator.add: operator.add,
    operator.and_: operator.and_,
    operator.eq: operator.eq,
    operator.floordiv: operator.floordiv,
    operator.ge: operator.ge,
    operator.gt: operator.gt,
    operator.le: operator.le,
    operator.lshift: operator.lshift,
    operator.lt: operator.lt,
    operator.mod: operator.mod,
    operator.mul: operator.mul,
    operator.ne: operator.ne,
    operator.or_: operator.or_,
    operator.rshift: operator.rshift,
    operator.sub: operator.sub,
    operator.truediv: operator.truediv,
    operator.xor: operator.xor,
    torch.sym_max: max,
    torch.sym_min: min,
}


class TridentFxImporter:
    def __init__(self, *, context: ir.Context, specialization_id: int):
        self.context = context
        self.specialization_id = specialization_id
        with context, ir.Location.unknown(context):
            self.module = ir.Module.create()
        self.values: dict[torch.fx.Node, Any] = {}
        self.symbols: dict[Any, ir.Value] = {}

    def argument(self, value: Any) -> ir.Value:
        if isinstance(value, torch.fx.Node):
            imported = self.values[value]
            if isinstance(imported, ir.Value):
                return imported
            if isinstance(imported, (tuple, list)):
                return array([box(element) for element in imported])
            raise NotImplementedError(f"aggregate node requires getitem: {value}")
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return self.symbolic(value.node.expr)
        if isinstance(value, (tuple, list)):
            return array([self.argument(element) for element in value])
        return constant(value)

    def symbolic(self, expression: Any) -> ir.Value:
        import sympy

        if expression in self.symbols:
            return self.symbols[expression]
        if expression.is_Integer:
            return native_constant(operator.index(expression))
        if expression.is_Float:
            return native_constant(float(expression))
        operations = {
            sympy.Add: operator.add,
            sympy.Mul: operator.mul,
            sympy.Min: min,
            sympy.Max: max,
        }
        if expression.func in operations:
            return reduce(
                lambda lhs, rhs: binary(operations[expression.func], lhs, rhs),
                (self.symbolic(argument) for argument in expression.args),
            )
        if expression.func.__name__ in ("FloorDiv", "Mod", "PythonMod"):
            lhs, rhs = (self.symbolic(argument) for argument in expression.args)
            operation = (
                operator.floordiv
                if expression.func.__name__ == "FloorDiv"
                else operator.mod
            )
            return binary(operation, lhs, rhs)
        if (
            expression.func is sympy.Pow
            and expression.args[1].is_Integer
            and expression.args[1] >= 0
        ):
            base = self.symbolic(expression.args[0])
            result = native_constant(1)
            for _ in range(operator.index(expression.args[1])):
                result = arith.muli(result, base)
            return result
        raise NotImplementedError(
            f"unbound symbolic FX expression: {expression}; refusing to use a concrete hint"
        )

    def _bind(self, node: torch.fx.Node, value: Any) -> None:
        self.values[node] = value
        metadata = node.meta.get("val")
        if isinstance(
            metadata, (torch.SymInt, torch.SymFloat, torch.SymBool)
        ) and isinstance(value, ir.Value):
            self.symbols[metadata.node.expr] = native(value)
        if isinstance(metadata, torch.Tensor) and isinstance(value, ir.Value):
            # Bind bare input shape symbols to runtime metadata. Composite
            # expressions are evaluated from their symbols, never from hints.
            for index, size in enumerate(metadata.shape):
                if (
                    isinstance(size, torch.SymInt)
                    and size.node.expr.is_Symbol
                    and size.node.expr not in self.symbols
                ):
                    self.symbols[size.node.expr] = ffi.tensor_size(
                        value, native_constant(index)
                    )

    def import_program(
        self, program: torch.export.ExportedProgram, *, func_name: str
    ) -> func.FuncOp:
        placeholders = {
            node.name: node for node in program.graph.nodes if node.op == "placeholder"
        }
        inputs = [
            spec
            for spec in program.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
            and not isinstance(spec.arg, ConstantArgument)
        ]
        outputs = [
            spec
            for spec in program.graph_signature.output_specs
            if spec.kind == OutputKind.USER_OUTPUT
        ]
        producers = {node.name: node for node in program.graph.nodes}
        with (
            self.context,
            ir.Location.unknown(self.context),
            ir.InsertionPoint(self.module.body),
        ):
            input_types = [
                value_type(placeholders[spec.arg.name].meta["val"]) for spec in inputs
            ]
            result_types = [
                value_type(
                    spec.arg.value
                    if isinstance(spec.arg, ConstantArgument)
                    else producers[spec.arg.name].meta["val"]
                )
                for spec in outputs
            ]
            function = func.FuncOp(
                func_name,
                ir.FunctionType.get(input_types, result_types),
                visibility="private",
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                for spec, value in zip(inputs, block.arguments, strict=True):
                    self._bind(placeholders[spec.arg.name], value)
                for spec in program.graph_signature.input_specs:
                    node = placeholders[spec.arg.name]
                    if spec.kind == InputKind.USER_INPUT:
                        if isinstance(spec.arg, ConstantArgument):
                            self.values[node] = constant(spec.arg.value)
                    else:
                        state = (
                            program.state_dict
                            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
                            else program.constants
                        )
                        self._bind(node, constant(state[spec.target]))
                self._import_nodes(program.graph, program.graph_module)
                results = [
                    box(
                        constant(spec.arg.value)
                        if isinstance(spec.arg, ConstantArgument)
                        else self.argument(producers[spec.arg.name])
                    )
                    for spec in outputs
                ]
                results = [
                    value if value.type == type_ else ffi.cast(type_, value)
                    for value, type_ in zip(results, result_types, strict=True)
                ]
                func.ReturnOp(results)
        return function

    def _import_nodes(
        self, graph: torch.fx.Graph, graph_module: torch.fx.GraphModule
    ) -> None:
        for node in graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op == "get_attr":
                value = graph_module
                for attribute in str(node.target).split("."):
                    value = getattr(value, attribute)
                self.values[node] = (
                    value
                    if isinstance(value, torch.fx.GraphModule)
                    else constant(value)
                )
                continue
            if node.op != "call_function":
                raise NotImplementedError(f"unsupported FX node: {node.format_node()}")
            target = node.target
            if target is operator.getitem:
                aggregate, index = node.args
                imported = self.values[aggregate]
                result = (
                    array_get(imported, index, value_type(node.meta["val"]))
                    if isinstance(imported, ir.Value)
                    else imported[index]
                )
            elif target in _SCALAR_OPERATIONS:
                lhs, rhs = (self.argument(argument) for argument in node.args)
                result = binary(_SCALAR_OPERATIONS[target], lhs, rhs)
            elif target is torch.sym_sum:
                arguments = (
                    node.args[0]
                    if len(node.args) == 1 and isinstance(node.args[0], (tuple, list))
                    else node.args
                )
                result = reduce(
                    arith.addi,
                    (native(self.argument(arg)) for arg in arguments),
                    native_constant(0),
                )
            elif target is operator.pow:
                base, exponent = node.args
                if not isinstance(exponent, int) or exponent < 0:
                    raise NotImplementedError(
                        "symbolic integer power requires a nonnegative constant exponent"
                    )
                result = native_constant(1)
                for _ in range(exponent):
                    result = arith.muli(result, native(self.argument(base)))
            elif target in (operator.neg, operator.not_, torch.sym_not):
                value = native(self.argument(node.args[0]))
                result = (
                    arith.xori(value, native_constant(True))
                    if target is not operator.neg
                    else binary(operator.sub, native_constant(0), value)
                )
            elif isinstance(target, torch._ops.OpOverload):
                result = self._aten(node)
            elif isinstance(target, torch._ops.HigherOrderOperator):
                if target.__name__ in (
                    "triton_kernel_wrapper_mutation",
                    "triton_kernel_wrapper_functional",
                ):
                    import_kernel(self, ir.Location.current, node)
                    continue
                if target.__name__ == "cond":
                    result = self._cond(node)
                else:
                    raise NotImplementedError(
                        f"unsupported higher-order FX operation: {target}"
                    )
            else:
                raise NotImplementedError(f"unsupported FX call: {target}")
            self._bind(node, result)

    def _aten(self, node: torch.fx.Node) -> Any:
        target = node.target
        schema = target._schema
        # FX may record the Tensor overload with a Python scalar operand.
        # Python dispatch accepts this shorthand; the boxed C++ schema does not.
        scalar_arguments = [
            index
            for index, argument in enumerate(node.args)
            if index < len(schema.arguments)
            and schema.arguments[index].type.kind() == "TensorType"
            and isinstance(
                argument.meta.get("val")
                if isinstance(argument, torch.fx.Node)
                else argument,
                (bool, int, float, torch.SymInt, torch.SymFloat, torch.SymBool),
            )
        ]
        if scalar_arguments and "Scalar" in target.overloadpacket.overloads():
            candidate = target.overloadpacket.Scalar._schema
            if all(
                index < len(candidate.arguments)
                and candidate.arguments[index].name == schema.arguments[index].name
                and candidate.arguments[index].type.kind()
                in ("NumberType", "IntType", "FloatType", "BoolType")
                for index in scalar_arguments
            ):
                schema = candidate
        name = schema.name.removeprefix("aten::")
        overload = schema.overload_name
        metadata_ops = {
            "sym_size": ffi.tensor_size,
            "sym_stride": ffi.tensor_stride,
            "sym_storage_offset": ffi.tensor_storage_offset,
            "dim": ffi.tensor_dim,
        }
        if name in metadata_ops:
            arguments = [native(self.argument(argument)) for argument in node.args]
            return metadata_ops[name](*arguments)
        if name == "sym_numel":
            tensor = self.argument(node.args[0])
            return reduce(
                arith.muli,
                (
                    ffi.tensor_size(tensor, native_constant(index))
                    for index in range(node.args[0].meta["val"].dim())
                ),
                native_constant(1),
            )
        if name == "_assert_scalar":
            from trident.core.dialects import cf

            cf.AssertOp(
                native(self.argument(node.args[0])),
                node.args[1] if len(node.args) > 1 else "FX assertion failed",
            )
            return constant(None)
        arguments = []
        for index, parameter in enumerate(schema.arguments):
            if index < len(node.args):
                argument = node.args[index]
            elif parameter.name in node.kwargs:
                argument = node.kwargs[parameter.name]
            elif parameter.has_default_value():
                argument = parameter.default_value
            else:
                raise ValueError(f"missing argument {parameter.name} for {node.target}")
            if isinstance(argument, (torch.layout, torch.memory_format)):
                enums = {
                    torch.strided: 0,
                    torch.sparse_coo: 1,
                    torch.sparse_csr: 2,
                    torch.contiguous_format: 0,
                    torch.preserve_format: 1,
                    torch.channels_last: 2,
                    torch.channels_last_3d: 3,
                }
                argument = enums[argument]
            imported = self.argument(argument)
            if isinstance(argument, torch.dtype):
                imported = checked_call(
                    "trident.runtime.tvm_ffi_to_torch_type", [imported], ffi_type("int")
                )
            arguments.append(imported)
        metadata = node.meta.get("val")
        result_type = value_type(metadata)
        registry_name = f"trident.aten.{name}" + (f".{overload}" if overload else "")
        result = checked_call(registry_name, arguments, result_type)
        if len(schema.returns) > 1:
            return tuple(
                array_get(result, index, value_type(element))
                for index, element in enumerate(metadata)
            )
        return result

    def _cond(self, node: torch.fx.Node) -> tuple[ir.Value, ...]:
        predicate, true_graph, false_graph, operands = node.args
        metadata = node.meta["val"]
        result_types = [value_type(element) for element in metadata]
        conditional = scf.IfOp(
            native(self.argument(predicate)), result_types, has_else=True
        )
        for block, graph in (
            (conditional.then_block, true_graph),
            (conditional.else_block, false_graph),
        ):
            child = self.values[graph]
            with ir.InsertionPoint(block):
                importer = TridentFxImporter(
                    context=self.context, specialization_id=self.specialization_id
                )
                importer.module = self.module
                placeholders = [n for n in child.graph.nodes if n.op == "placeholder"]
                for placeholder, operand in zip(placeholders, operands, strict=True):
                    importer._bind(placeholder, self.argument(operand))
                importer._import_nodes(child.graph, child)
                output = next(n for n in child.graph.nodes if n.op == "output")
                scf.YieldOp([box(importer.argument(value)) for value in output.args[0]])
        return tuple(conditional.results)
