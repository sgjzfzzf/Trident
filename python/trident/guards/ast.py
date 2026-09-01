# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import warnings
from collections.abc import Callable
from functools import reduce
from itertools import chain, pairwise
from typing import TypeAlias

from trident.core import ir
from trident.core.dialects import arith, tvm_ffi
from trident.input import InputTable

from .local import Local

GuardBuildFn: TypeAlias = Callable[[InputTable, ir.Context], ir.Value]
TensorMetadataFn: TypeAlias = Callable[[ir.Type, ir.Value], ir.Value]
TensorIndexedMetadataFn: TypeAlias = Callable[[ir.Type, ir.Value, ir.Value], ir.Value]


class _SkipBaseGuard(Exception): ...


class ASTVisitor(ast.NodeVisitor):
    """Compose delayed IR builders from supported Dynamo guard AST nodes."""

    def __init__(self, text: str) -> None:
        self.text = text

    def _apply_binary(
        self,
        operation: Callable[[ir.Value, ir.Value], ir.Value],
        lhs: GuardBuildFn,
        rhs: GuardBuildFn,
    ) -> GuardBuildFn:
        return lambda tree, context: operation(
            self._value(lhs, tree, context),
            self._value(rhs, tree, context),
        )

    def _build_identity(self, lhs: Local, rhs: Local) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            lhs_value = lhs.resolve(tree)
            rhs_value = rhs.resolve(tree)
            assert lhs_value is not None and rhs_value is not None, (
                f"guard source cannot be resolved: {self.text!r}"
            )
            tensor_type_ids = {
                ir.Type.parse(type_text, context=context).typeid
                for type_text in (
                    "!torch.tensor",
                    "!torch.vtensor",
                    "!tvm_ffi.tensor",
                )
            }
            if not all(
                value.type.typeid in tensor_type_ids for value in (lhs_value, rhs_value)
            ):
                i1 = ir.IntegerType.get_signless(1, context)
                return tvm_ffi.eq(i1, lhs_value, rhs_value)
            warnings.warn(
                "Skipping unsupported Tensor identity guard; object identity "
                f"will not be validated: {self.text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._true(context)

        return build

    def _build_not_sequence(self, source: Local) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            value = source.resolve(tree)
            assert value is not None, f"guard source cannot be resolved: {self.text!r}"
            i64 = ir.IntegerType.get_signless(64, context)
            return arith.cmpi(
                arith.CmpIPredicate.eq,
                tvm_ffi.array_length(i64, value),
                arith.constant(i64, ir.IntegerAttr.get(i64, 0)),
            )

        return build

    def _build_tensor_indexed_metadata(
        self,
        source: Local,
        operation: TensorIndexedMetadataFn,
        index: int,
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            tensor = source.resolve(tree)
            assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
            i64 = ir.IntegerType.get_signless(64, context)
            dimension = arith.constant(i64, ir.IntegerAttr.get(i64, index))
            return operation(i64, tensor, dimension)

        return build

    def _build_tensor_metadata(
        self,
        source: Local,
        operation: TensorMetadataFn,
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            tensor = source.resolve(tree)
            assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
            i64 = ir.IntegerType.get_signless(64, context)
            return operation(i64, tensor)

        return build

    @staticmethod
    def _constant(value: int) -> GuardBuildFn:
        return lambda _, context: arith.constant(
            (i64 := ir.IntegerType.get_signless(64, context)),
            ir.IntegerAttr.get(i64, value),
        )

    @staticmethod
    def _constant_i1(value: bool) -> GuardBuildFn:
        return lambda _, context: arith.constant(
            (i1 := ir.IntegerType.get_signless(1, context)),
            ir.IntegerAttr.get(i1, int(value)),
        )

    def _error(self, message: str) -> GuardBuildFn:
        def build(_: InputTable, __: ir.Context) -> ir.Value:
            assert False, f"{message}: {self.text!r}"

        return build

    def _skip_base(self) -> GuardBuildFn:
        def build(_: InputTable, __: ir.Context) -> ir.Value:
            warnings.warn(
                "Skipping unsupported Tensor._base guard; view metadata will "
                f"not be validated: {self.text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            raise _SkipBaseGuard

        return build

    @staticmethod
    def _true(context: ir.Context) -> ir.Value:
        i1 = ir.IntegerType.get_signless(1, context)
        return arith.constant(i1, ir.IntegerAttr.get(i1, 1))

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
        operations = {
            "ndimension": tvm_ffi.tensor_dim,
            "storage_offset": tvm_ffi.tensor_storage_offset,
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
            except _SkipBaseGuard:
                return self._true(context)

        return build

    def generic_visit(self, node: ast.AST) -> None:
        return None

    def visit_Attribute(self, node: ast.Attribute) -> GuardBuildFn | None:
        if node.attr == "_base":
            return self._skip_base()
        if isinstance(node.value, ast.Attribute):
            return self.visit(node.value)
        return None

    def visit_BinOp(self, node: ast.BinOp) -> GuardBuildFn | None:
        operations: dict[
            type[ast.operator], Callable[[ir.Value, ir.Value], ir.Value]
        ] = {
            ast.Add: arith.addi,
            ast.Mult: arith.muli,
            ast.BitOr: arith.ori,
            ast.RShift: arith.shrsi,
            ast.Sub: arith.subi,
            ast.FloorDiv: arith.floordivsi,
            # Modulo guards operate on positive shape values.
            ast.Mod: arith.remsi,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            return None
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if lhs is None or rhs is None:
            return None
        return self._apply_binary(operation, lhs, rhs)

    def visit_BoolOp(self, node: ast.BoolOp) -> GuardBuildFn | None:
        values = [self.visit(value) for value in node.values]
        if any(value is None for value in values):
            return None
        function = arith.andi if isinstance(node.op, ast.And) else arith.ori

        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            i1 = ir.IntegerType.get_signless(1, context)
            initial = arith.constant(
                i1,
                ir.IntegerAttr.get(i1, 1 if isinstance(node.op, ast.And) else 0),
            )
            return reduce(
                function,
                (self._value(value, tree, context) for value in values if value),
                initial,
            )

        return build

    def visit_Call(self, node: ast.Call) -> GuardBuildFn | None:
        operations: dict[str, Callable[[ir.Value, ir.Value], ir.Value]] = {
            "min": arith.minsi,
            "max": arith.maxsi,
        }
        if (
            isinstance(node.func, ast.Name)
            and len(node.args) == 2
            and not node.keywords
            and node.func.id in operations
        ):
            lhs = self.visit(node.args[0])
            rhs = self.visit(node.args[1])
            if lhs is None or rhs is None:
                return None
            return self._apply_binary(operations[node.func.id], lhs, rhs)

        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "len"
            or len(node.args) != 1
            or node.keywords
        ):
            return self._visit_method_call(node)

        argument = node.args[0]
        is_tensor_shape = (
            isinstance(argument, ast.Attribute) and argument.attr == "shape"
        )
        source = Local.from_expression(argument.value if is_tensor_shape else argument)
        if source is None:
            return None
        return self._build_tensor_metadata(
            source,
            tvm_ffi.tensor_dim if is_tensor_shape else tvm_ffi.array_length,
        )

    def visit_Compare(self, node: ast.Compare) -> GuardBuildFn | None:
        if any(isinstance(operation, ast.Is) for operation in node.ops):
            if len(node.ops) != 1:
                return self._error("identity guard comparison must be binary")
            [comparator] = node.comparators
            lhs_local = Local.from_expression(node.left)
            rhs_local = Local.from_expression(comparator)
            if lhs_local is None or rhs_local is None or lhs_local == rhs_local:
                return None
            return self._build_identity(lhs_local, rhs_local)

        predicates = [
            {
                ast.Eq: arith.CmpIPredicate.eq,
                ast.NotEq: arith.CmpIPredicate.ne,
                ast.Lt: arith.CmpIPredicate.slt,
                ast.LtE: arith.CmpIPredicate.sle,
                ast.Gt: arith.CmpIPredicate.sgt,
                ast.GtE: arith.CmpIPredicate.sge,
            }.get(type(operation))
            for operation in node.ops
        ]
        if any(predicate is None for predicate in predicates):
            return self._error("unsupported guard comparison operator")
        values = [
            self.visit(operand) for operand in chain((node.left,), node.comparators)
        ]
        if any(value is None for value in values):
            return self._error("guard comparison operand cannot be lowered")

        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            i1 = ir.IntegerType.get_signless(1, context)
            true = arith.constant(i1, ir.IntegerAttr.get(i1, 1))
            return reduce(
                arith.andi,
                (
                    arith.cmpi(
                        predicate,
                        self._value(lhs, tree, context),
                        self._value(rhs, tree, context),
                    )
                    for predicate, (lhs, rhs) in zip(
                        predicates, pairwise(values), strict=True
                    )
                ),
                true,
            )

        return build

    def visit_Constant(self, node: ast.Constant) -> GuardBuildFn | None:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return self._constant(node.value)
        return None

    def visit_IfExp(self, node: ast.IfExp) -> GuardBuildFn | None:
        condition = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        if condition is None or body is None or orelse is None:
            return None

        return lambda tree, context: arith.select(
            self._value(condition, tree, context),
            self._value(body, tree, context),
            self._value(orelse, tree, context),
        )

    def visit_Subscript(self, node: ast.Subscript) -> GuardBuildFn | None:
        source = Local.from_expression(node)
        if source is not None:

            def build(tree: InputTable, __: ir.Context) -> ir.Value:
                value = source.resolve(tree)
                assert value is not None, (
                    f"guard source cannot be resolved: {self.text!r}"
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
            "size": tvm_ffi.tensor_size,
            "stride": tvm_ffi.tensor_stride,
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
            return self._apply_binary(
                arith.xori,
                operand,
                self._constant_i1(True),
            )
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):

            def build(tree: InputTable, context: ir.Context) -> ir.Value:
                i64 = ir.IntegerType.get_signless(64, context)
                zero = arith.constant(i64, ir.IntegerAttr.get(i64, 0))
                return arith.subi(zero, self._value(operand, tree, context))

            return build
        return None
