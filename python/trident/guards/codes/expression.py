# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from collections.abc import Hashable
from functools import reduce
from typing import Final, Self, override

from trident.core import ir
from trident.core.dialects import arith, tvm_ffi

from ..local import Local, SourceTree
from .base import GuardCode


class ExpressionVisitor(ast.NodeVisitor):
    def __init__(
        self,
        tree: SourceTree,
        context: ir.Context,
        text: str,
    ) -> None:
        self.tree = tree
        self.context = context
        self.text = text

    def visit(self, node: ast.AST) -> ir.Value | None:
        return super().visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        return None

    def visit_Constant(self, node: ast.Constant) -> ir.Value | None:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            i64 = ir.IntegerType.get_signless(64, self.context)
            return arith.constant(
                i64,
                ir.IntegerAttr.get(i64, node.value),
            )
        else:
            return None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.Value | None:
        value = self.visit(node.operand)
        if value is None:
            return None
        elif isinstance(node.op, ast.UAdd):
            return value
        elif isinstance(node.op, ast.USub):
            i64 = ir.IntegerType.get_signless(64, self.context)
            zero = arith.constant(i64, ir.IntegerAttr.get(i64, 0))
            return arith.subi(zero, value)
        elif isinstance(node.op, ast.Not):
            i1 = ir.IntegerType.get_signless(1, self.context)
            true = arith.constant(i1, ir.IntegerAttr.get(i1, 1))
            return arith.xori(value, true)
        else:
            return None

    def visit_BinOp(self, node: ast.BinOp) -> ir.Value | None:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if lhs is None or rhs is None:
            return None
        elif isinstance(node.op, ast.Add):
            return arith.addi(lhs, rhs)
        elif isinstance(node.op, ast.Sub):
            return arith.subi(lhs, rhs)
        elif isinstance(node.op, ast.Mult):
            return arith.muli(lhs, rhs)
        elif isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            if not (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)
                and not isinstance(node.right.value, bool)
                and node.right.value != 0
            ):
                return None
            else:
                quotient = self.floor_div(lhs, rhs)
                if isinstance(node.op, ast.FloorDiv):
                    return quotient
                else:
                    return arith.subi(lhs, arith.muli(quotient, rhs))
        else:
            return None

    def visit_Compare(self, node: ast.Compare) -> ir.Value | None:
        lhs = self.visit(node.left)
        if lhs is None:
            return None
        comparisons: list[ir.Value] = []
        for operation, comparator in zip(node.ops, node.comparators):
            rhs = self.visit(comparator)
            if rhs is None:
                return None
            predicate = {
                ast.Eq: arith.CmpIPredicate.eq,
                ast.NotEq: arith.CmpIPredicate.ne,
                ast.Lt: arith.CmpIPredicate.slt,
                ast.LtE: arith.CmpIPredicate.sle,
                ast.Gt: arith.CmpIPredicate.sgt,
                ast.GtE: arith.CmpIPredicate.sge,
            }.get(type(operation))
            if predicate is None:
                return None
            comparisons.append(arith.cmpi(predicate, lhs, rhs))
            lhs = rhs
        i1 = ir.IntegerType.get_signless(1, self.context)
        true = arith.constant(
            i1,
            ir.IntegerAttr.get(i1, 1),
        )
        return reduce(arith.andi, comparisons, true)

    def visit_BoolOp(self, node: ast.BoolOp) -> ir.Value | None:
        values = [self.visit(value) for value in node.values]
        if any(value is None for value in values):
            return None
        i1 = ir.IntegerType.get_signless(1, self.context)
        init = arith.constant(
            i1,
            ir.IntegerAttr.get(i1, int(isinstance(node.op, ast.And))),
        )
        function = arith.andi if isinstance(node.op, ast.And) else arith.ori
        return reduce(
            function,
            [value for value in values if value is not None],
            init,
        )

    def visit_Subscript(self, node: ast.Subscript) -> ir.Value | None:
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
        source = self.local(call.func.value)
        if source is None:
            return None
        tensor = source.resolve(self.tree)
        if tensor is None:
            return None
        i64 = ir.IntegerType.get_signless(64, self.context)
        index = arith.constant(
            i64,
            ir.IntegerAttr.get(i64, node.slice.value),
        )
        if call.func.attr == "size":
            return tvm_ffi.tensor_size(i64, tensor, index)
        if call.func.attr == "stride":
            return tvm_ffi.tensor_stride(i64, tensor, index)
        return None

    def visit_Call(self, node: ast.Call) -> ir.Value | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        if node.args or node.keywords:
            return None
        source = self.local(node.func.value)
        if source is None:
            return None
        tensor = source.resolve(self.tree)
        if tensor is None:
            return None
        i64 = ir.IntegerType.get_signless(64, self.context)
        if node.func.attr == "storage_offset":
            return tvm_ffi.tensor_storage_offset(i64, tensor)
        if node.func.attr == "ndimension":
            return tvm_ffi.tensor_dim(i64, tensor)
        return None

    def floor_div(self, lhs: ir.Value, rhs: ir.Value) -> ir.Value | None:
        i64 = ir.IntegerType.get_signless(64, self.context)

        def constant(value: int) -> ir.Value:
            return arith.constant(
                i64,
                ir.IntegerAttr.get(i64, value),
            )

        truncated = arith.divsi(lhs, rhs)
        remainder = arith.remsi(lhs, rhs)
        non_zero = arith.cmpi(
            arith.CmpIPredicate.ne,
            remainder,
            constant(0),
        )
        lhs_negative = arith.cmpi(
            arith.CmpIPredicate.slt,
            lhs,
            constant(0),
        )
        rhs_negative = arith.cmpi(
            arith.CmpIPredicate.slt,
            rhs,
            constant(0),
        )
        signs_differ = arith.xori(lhs_negative, rhs_negative)
        adjust = arith.andi(non_zero, signs_differ)
        decremented = arith.subi(truncated, constant(1))
        return arith.select(adjust, decremented, truncated)

    def local(self, expression: ast.expr) -> Local | None:
        return Local.from_expression(expression)


class ExpressionCode(GuardCode):
    priority = 2

    def __init__(self, text: str, expression: ast.expr) -> None:
        super().__init__(text, None)
        self.expression: Final[ast.expr] = expression

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self:
        return cls(text, ast.parse(text, mode="eval").body)

    @property
    def key(self) -> Hashable:
        return ("shape", ast.dump(self.expression, include_attributes=False))

    @override
    def build(
        self,
        tree: SourceTree,
        context: ir.Context,
    ) -> ir.Value:
        result = ExpressionVisitor(tree, context, self.text).visit(self.expression)
        return result if result is not None else super().build(tree, context)
