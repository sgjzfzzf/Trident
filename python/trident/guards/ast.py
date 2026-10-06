# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import operator
import warnings
from collections.abc import Callable
from functools import reduce
from itertools import chain, pairwise
from typing import Any, ClassVar, TypeAlias

import torch

from trident.core import ir
from trident.core.dialects import arith
from trident.core.dialects import tvm_ffi as ffi
from trident.input import InputTable
from trident.ir_utils import (
    binary,
    checked_call,
    constant,
    equal,
    ffi_type,
    native,
    native_constant,
)

from .local import Local

GuardBuildFn: TypeAlias = Callable[[InputTable, ir.Context], ir.Value]
TensorMetadataFn: TypeAlias = Callable[[ir.Value], ir.Value]
TensorIndexedMetadataFn: TypeAlias = Callable[[ir.Value, ir.Value], ir.Value]
ScalarOperationKey: TypeAlias = type[ast.operator | ast.cmpop] | str


class _SkipGuard(Exception):
    def __init__(self, result: bool):
        super().__init__(result)
        self.result = result


class ASTVisitor(ast.NodeVisitor):
    """Compose delayed native/TVM FFI builders from Dynamo guard expressions."""

    _scalar_ops: ClassVar[dict[ScalarOperationKey, Callable[..., Any]]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.BitOr: operator.or_,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        "min": min,
        "max": max,
    }

    def __init__(self, text: str) -> None:
        self.text = text

    @staticmethod
    def _build_binary_template(
        lhs: GuardBuildFn, rhs: GuardBuildFn, operation_key: ScalarOperationKey
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            return binary(
                ASTVisitor._scalar_ops[operation_key],
                lhs(tree, context),
                rhs(tree, context),
            )

        return build

    def _build_comparison(
        self, operation: ast.cmpop, lhs: GuardBuildFn, rhs: GuardBuildFn
    ) -> GuardBuildFn:
        if isinstance(operation, ast.Eq):
            return lambda tree, context: equal(lhs(tree, context), rhs(tree, context))
        return self._build_binary_template(lhs, rhs, type(operation))

    def _build_identity(self, lhs: Local, rhs: Local) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            lhs_value, rhs_value = lhs.resolve(tree), rhs.resolve(tree)
            assert lhs_value is not None and rhs_value is not None, (
                f"guard source cannot be resolved: {self.text!r}"
            )
            if (
                str(lhs_value.type) != "!tvm_ffi.tensor"
                or str(rhs_value.type) != "!tvm_ffi.tensor"
            ):
                return equal(lhs_value, rhs_value)
            warnings.warn(
                f"Skipping unsupported Tensor identity guard; object identity will not be validated: {self.text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return native_constant(True)

        return build

    @staticmethod
    def _build_logical_and(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return arith.andi(native(lhs), native(rhs))

    @staticmethod
    def _build_logical_not(value: ir.Value) -> ir.Value:
        return arith.xori(native(value), native_constant(True))

    @staticmethod
    def _build_logical_or(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return arith.ori(native(lhs), native(rhs))

    @staticmethod
    def _build_negate(value: ir.Value) -> ir.Value:
        value = native(value)
        return (
            arith.negf(value)
            if str(value.type) == "f64"
            else arith.subi(native_constant(0), value)
        )

    def _build_not_sequence(self, source: Local) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            value = source.resolve(tree)
            assert value is not None, f"guard source cannot be resolved: {self.text!r}"
            return binary(operator.eq, self._sequence_length(value), native_constant(0))

        return build

    @staticmethod
    def _sequence_length(value: ir.Value) -> ir.Value:
        return native(checked_call("ffi.ArraySize", [value], ffi_type("int")))

    @staticmethod
    def _build_select(
        condition: ir.Value, true_value: ir.Value, false_value: ir.Value
    ) -> ir.Value:
        return arith.select(native(condition), native(true_value), native(false_value))

    def _build_tensor_indexed_metadata(
        self, source: Local, operation: TensorIndexedMetadataFn, index: int
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            tensor = source.resolve(tree)
            assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
            return operation(tensor, native_constant(index))

        return build

    def _build_tensor_metadata(
        self, source: Local, operation: TensorMetadataFn
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            tensor = source.resolve(tree)
            assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
            return operation(tensor)

        return build

    @staticmethod
    def _constant(value: Any) -> GuardBuildFn | None:
        if value is None or isinstance(
            value, (torch.device, torch.dtype, bool, int, float, str)
        ):
            return lambda _, context: constant(value)
        return None

    def _error(self, message: str) -> GuardBuildFn:
        def build(_: InputTable, context: ir.Context) -> ir.Value:
            assert False, f"{message}: {self.text!r} in {context}"

        return build

    def _skip_base(self) -> GuardBuildFn:
        def build(_: InputTable, context: ir.Context) -> ir.Value:
            warnings.warn(
                f"Unsupported Tensor._base guard forces a specialization miss: {self.text!r} in {context}",
                RuntimeWarning,
                stacklevel=2,
            )
            raise _SkipGuard(False)

        return build

    def _skip_storage_offset(self) -> GuardBuildFn:
        def build(_: InputTable, __: ir.Context) -> ir.Value:
            # Python Tensor arguments reach the packed TVM FFI function via
            # DLPack. That boundary exposes the logical data pointer and
            # canonicalizes byte_offset to zero, so the original PyTorch
            # storage_offset is neither observable nor needed for addressing.
            raise _SkipGuard(True)

        return build

    @staticmethod
    def _false(context: ir.Context) -> ir.Value:
        return native_constant(False)

    @staticmethod
    def _true(context: ir.Context) -> ir.Value:
        return native_constant(True)

    @staticmethod
    def _value(
        build_fn: GuardBuildFn,
        tree: InputTable,
        context: ir.Context,
    ) -> ir.Value:
        return build_fn(tree, context)

    def _visit_method_call(self, node: ast.Call) -> GuardBuildFn | None:
        if not isinstance(node.func, ast.Attribute) or node.args or node.keywords:
            return None
        if (
            isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_base"
        ):
            return self._skip_base()
        source = Local.from_expression(node.func.value)
        if source is None:
            return None
        if node.func.attr == "storage_offset":
            return self._skip_storage_offset()
        operations = {
            "ndimension": ffi.tensor_dim,
        }
        operation = operations.get(node.func.attr)
        return self._build_tensor_metadata(source, operation) if operation else None

    def build(self, expression: ast.expr | str) -> GuardBuildFn | None:
        if isinstance(expression, str):
            expression = ast.parse(expression, mode="eval").body
        build_fn = self.visit(expression)
        if build_fn is None:
            return None

        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            try:
                return build_fn(tree, context)
            except _SkipGuard as skip:
                return self._true(context) if skip.result else self._false(context)

        return build

    def generic_visit(self, node: ast.AST) -> None:
        return None

    def visit_Attribute(self, node: ast.Attribute) -> GuardBuildFn | None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "torch"
            and isinstance(value := getattr(torch, node.attr, None), torch.dtype)
        ):
            return self._constant(value)
        if node.attr == "_base":
            return self._skip_base()
        if isinstance(node.value, ast.Attribute):
            return self.visit(node.value)
        return None

    def visit_BinOp(self, node: ast.BinOp) -> GuardBuildFn | None:
        if type(node.op) not in self._scalar_ops:
            return None
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if lhs is None or rhs is None:
            return None
        return self._build_binary_template(
            lhs,
            rhs,
            type(node.op),
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> GuardBuildFn | None:
        values = [self.visit(value) for value in node.values]
        if any(value is None for value in values):
            return None
        function = (
            self._build_logical_and
            if isinstance(node.op, ast.And)
            else self._build_logical_or
        )

        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            return reduce(
                function,
                (self._value(value, tree, context) for value in values if value),
            )

        return build

    def visit_Call(self, node: ast.Call) -> GuardBuildFn | None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "device"
            and not node.args
        ):
            try:
                keywords = {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in node.keywords
                }
                if len(keywords) != len(node.keywords):
                    return None
                value = torch.device(**keywords)
            except (SyntaxError, TypeError, ValueError, RuntimeError):
                return None
            return self._constant(value)

        operations = {"min", "max"}
        if (
            isinstance(node.func, ast.Name)
            and len(node.args) == 2
            and not node.keywords
            and node.func.id in operations
        ):
            lhs_node, rhs_node = node.args
            lhs = self.visit(lhs_node)
            rhs = self.visit(rhs_node)
            if lhs is None or rhs is None:
                return None
            return self._build_binary_template(
                lhs,
                rhs,
                node.func.id,
            )

        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "len"
            or len(node.args) != 1
            or node.keywords
        ):
            return self._visit_method_call(node)

        [argument] = node.args
        is_tensor_shape = (
            isinstance(argument, ast.Attribute) and argument.attr == "shape"
        )
        source = Local.from_expression(argument.value if is_tensor_shape else argument)
        if source is None:
            return None
        return self._build_tensor_metadata(
            source,
            ffi.tensor_dim if is_tensor_shape else self._sequence_length,
        )

    def visit_Compare(self, node: ast.Compare) -> GuardBuildFn | None:
        if any(isinstance(operation, ast.Is) for operation in node.ops):
            if len(node.ops) != 1:
                return self._error("identity guard comparison must be binary")
            [comparator] = node.comparators
            if isinstance(comparator, ast.Constant) and comparator.value is None:
                return self.visit_Compare(
                    ast.Compare(node.left, [ast.Eq()], [comparator])
                )
            lhs_local = Local.from_expression(node.left)
            rhs_local = Local.from_expression(comparator)
            if lhs_local is None or rhs_local is None or lhs_local == rhs_local:
                return None
            return self._build_identity(lhs_local, rhs_local)

        if any(
            not isinstance(
                operation,
                (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE),
            )
            for operation in node.ops
        ):
            return self._error("unsupported guard comparison operator")
        values = [
            self.visit(operand) for operand in chain((node.left,), node.comparators)
        ]
        if any(value is None for value in values):
            return self._error("guard comparison operand cannot be lowered")
        comparison_builders = tuple(
            self._build_comparison(operation, lhs, rhs)
            for operation, (lhs, rhs) in zip(node.ops, pairwise(values), strict=True)
        )

        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            return reduce(
                self._build_logical_and,
                (comparison(tree, context) for comparison in comparison_builders),
                native_constant(True),
            )

        return build

    def visit_Constant(self, node: ast.Constant) -> GuardBuildFn | None:
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return self._constant(node.value)
        return None

    def visit_IfExp(self, node: ast.IfExp) -> GuardBuildFn | None:
        condition = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        if condition is None or body is None or orelse is None:
            return None

        return lambda tree, context: self._build_select(
            self._value(condition, tree, context),
            self._value(body, tree, context),
            self._value(orelse, tree, context),
        )

    def visit_Subscript(self, node: ast.Subscript) -> GuardBuildFn | None:
        source = Local.from_expression(node)
        if source is not None:

            def build(tree: InputTable, context: ir.Context) -> ir.Value:
                value = source.resolve(tree)
                assert value is not None, (
                    f"guard source cannot be resolved: {self.text!r} in {context}"
                )
                return value

            return build
        if not isinstance(node.value, ast.Call):
            return None
        call = node.value
        if not (
            isinstance(call.func, ast.Attribute)
            and not call.args
            and not call.keywords
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
            and not isinstance(node.slice.value, bool)
        ):
            return None
        source = Local.from_expression(call.func.value)
        if (
            source is None
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "_base"
        ):
            return self._skip_base()
        if source is None:
            return None
        operations = {
            "size": ffi.tensor_size,
            "stride": ffi.tensor_stride,
        }
        operation = operations.get(call.func.attr)
        if operation is None:
            return None

        return self._build_tensor_indexed_metadata(
            source,
            operation,
            node.slice.value,
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> GuardBuildFn | None:
        if isinstance(node.op, ast.Not):
            source = Local.from_expression(node.operand)
            if source is not None:
                return self._build_not_sequence(source)
        operand = self.visit(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.Not):
            return lambda tree, context: self._build_logical_not(
                self._value(operand, tree, context),
            )
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return lambda tree, context: self._build_negate(
                self._value(operand, tree, context),
            )
        return None
