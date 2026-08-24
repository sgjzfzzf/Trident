# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import warnings
from collections.abc import Hashable
from functools import reduce
from itertools import chain, pairwise
from typing import Final, Self, override

from trident.core import ir
from trident.core.dialects import arith, tvm_ffi

from ..local import Local, SourceTree
from .base import GuardCode


class _SkipBaseGuard(Exception): ...


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

    def build(self, expression: ast.expr) -> ir.Value | None:
        try:
            return self.visit(expression)
        except _SkipBaseGuard:
            i1 = ir.IntegerType.get_signless(1, self.context)
            return arith.constant(i1, ir.IntegerAttr.get(i1, 1))

    def visit_Attribute(self, node: ast.Attribute) -> ir.Value | None:
        if node.attr == "_base":
            warnings.warn(
                "Skipping unsupported Tensor._base guard; view metadata will "
                f"not be validated: {self.text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            raise _SkipBaseGuard
        elif isinstance(node.value, ast.Attribute):
            return self.visit(node.value)
        else:
            return None

    def visit_BinOp(self, node: ast.BinOp) -> ir.Value | None:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if lhs is None or rhs is None:
            return None
        elif isinstance(node.op, ast.Add):
            return arith.addi(lhs, rhs)
        elif isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            if (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)
                and not isinstance(node.right.value, bool)
                and node.right.value != 0
            ):
                i64 = ir.IntegerType.get_signless(64, self.context)
                zero = arith.constant(i64, ir.IntegerAttr.get(i64, 0))
                one = arith.constant(i64, ir.IntegerAttr.get(i64, 1))

                truncated = arith.divsi(lhs, rhs)
                remainder = arith.remsi(lhs, rhs)
                non_zero = arith.cmpi(
                    arith.CmpIPredicate.ne,
                    remainder,
                    zero,
                )
                lhs_negative = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    lhs,
                    zero,
                )
                rhs_negative = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    rhs,
                    zero,
                )
                signs_differ = arith.xori(lhs_negative, rhs_negative)
                adjust = arith.andi(non_zero, signs_differ)
                decremented = arith.subi(truncated, one)
                quotient = arith.select(adjust, decremented, truncated)
                if isinstance(node.op, ast.FloorDiv):
                    return quotient
                else:
                    return arith.subi(lhs, arith.muli(quotient, rhs))
            else:
                return None
        elif isinstance(node.op, ast.Mult):
            return arith.muli(lhs, rhs)
        elif isinstance(node.op, ast.Sub):
            return arith.subi(lhs, rhs)
        else:
            return None

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
            filter(None, values),
            init,
        )

    def visit_Call(self, node: ast.Call) -> ir.Value | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        # Custom visitors do not recurse into call attributes automatically.
        self.visit(node.func)
        if node.args or node.keywords:
            return None
        source = Local.from_expression(node.func.value)
        if source is None:
            return None
        tensor = source.resolve(self.tree)
        if tensor is None:
            return None
        i64 = ir.IntegerType.get_signless(64, self.context)
        if node.func.attr == "ndimension":
            return tvm_ffi.tensor_dim(i64, tensor)
        elif node.func.attr == "storage_offset":
            return tvm_ffi.tensor_storage_offset(i64, tensor)
        else:
            return None

    def visit_Compare(self, node: ast.Compare) -> ir.Value | None:
        if any(isinstance(operation, ast.Is) for operation in node.ops):
            assert len(node.ops) == 1, (
                f"identity guard comparison must be binary: {self.text!r}"
            )
            [comparator] = node.comparators
            lhs_local = Local.from_expression(node.left)
            rhs_local = Local.from_expression(comparator)
            if lhs_local is None or rhs_local is None or lhs_local == rhs_local:
                return None
            lhs_value = lhs_local.resolve(self.tree)
            rhs_value = rhs_local.resolve(self.tree)
            if lhs_value is None or rhs_value is None:
                return None
            tensor_type_ids = {
                ir.Type.parse(type_text, context=self.context).typeid
                for type_text in (
                    "!torch.tensor",
                    "!torch.vtensor",
                    "!tvm_ffi.tensor",
                )
            }
            if not all(
                value.type.typeid in tensor_type_ids for value in (lhs_value, rhs_value)
            ):
                return tvm_ffi.eq(
                    ir.IntegerType.get_signless(1, self.context),
                    lhs_value,
                    rhs_value,
                )
            warnings.warn(
                "Skipping unsupported Tensor identity guard; object identity "
                f"will not be validated: {self.text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            i1 = ir.IntegerType.get_signless(1, self.context)
            return arith.constant(i1, ir.IntegerAttr.get(i1, 1))
        else:
            predicates = [
                predicate
                for operation in node.ops
                if (
                    predicate := {
                        ast.Eq: arith.CmpIPredicate.eq,
                        ast.NotEq: arith.CmpIPredicate.ne,
                        ast.Lt: arith.CmpIPredicate.slt,
                        ast.LtE: arith.CmpIPredicate.sle,
                        ast.Gt: arith.CmpIPredicate.sgt,
                        ast.GtE: arith.CmpIPredicate.sge,
                    }.get(type(operation))
                )
                is not None
            ]
            assert len(predicates) == len(node.ops), (
                f"unsupported guard comparison operator: {self.text!r}"
            )
            values = [
                value
                for operand in chain((node.left,), node.comparators)
                if (value := self.visit(operand)) is not None
            ]
            assert len(values) == len(node.ops) + 1, (
                f"guard comparison operand cannot be lowered: {self.text!r}"
            )
            i1 = ir.IntegerType.get_signless(1, self.context)
            true = arith.constant(
                i1,
                ir.IntegerAttr.get(i1, 1),
            )
            return reduce(
                arith.andi,
                map(
                    lambda predicate, operands: arith.cmpi(predicate, *operands),
                    predicates,
                    pairwise(values),
                ),
                true,
            )

    def visit_Constant(self, node: ast.Constant) -> ir.Value | None:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            i64 = ir.IntegerType.get_signless(64, self.context)
            return arith.constant(
                i64,
                ir.IntegerAttr.get(i64, node.value),
            )
        else:
            return None

    def visit_Subscript(self, node: ast.Subscript) -> ir.Value | None:
        if not isinstance(node.value, ast.Call):
            return None
        call = node.value
        # Custom visitors do not recurse into call attributes automatically.
        self.visit(call.func)
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
        elif call.func.attr == "stride":
            return tvm_ffi.tensor_stride(i64, tensor, index)
        else:
            return None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.Value | None:
        value = self.visit(node.operand)
        if value is None:
            return None
        elif isinstance(node.op, ast.Not):
            i1 = ir.IntegerType.get_signless(1, self.context)
            true = arith.constant(i1, ir.IntegerAttr.get(i1, 1))
            return arith.xori(value, true)
        elif isinstance(node.op, ast.UAdd):
            return value
        elif isinstance(node.op, ast.USub):
            i64 = ir.IntegerType.get_signless(64, self.context)
            zero = arith.constant(i64, ir.IntegerAttr.get(i64, 0))
            return arith.subi(zero, value)
        else:
            return None

    def generic_visit(self, node: ast.AST) -> None:
        return None


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
    ) -> Self | None:
        try:
            expression = ast.parse(text, mode="eval").body
        except SyntaxError:
            return None
        if source is not None:
            if not (
                isinstance(expression, ast.Compare)
                and len(expression.ops) == 1
                and isinstance(expression.ops[0], ast.Is)
            ):
                return None
            lhs = Local.from_expression(expression.left)
            rhs = Local.from_expression(expression.comparators[0])
            if lhs is None or rhs is None or lhs == rhs or source not in (lhs, rhs):
                return None
        return cls(text, expression)

    @property
    def key(self) -> Hashable:
        return ("shape", ast.dump(self.expression, include_attributes=False))

    @override
    def build(
        self,
        tree: SourceTree,
        context: ir.Context,
    ) -> ir.Value:
        return ExpressionVisitor(tree, context, self.text).build(
            self.expression
        ) or super().build(tree, context)
