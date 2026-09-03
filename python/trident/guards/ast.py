# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import warnings
from collections.abc import Callable
from functools import reduce
from itertools import chain, pairwise
from typing import ClassVar, TypeAlias

from trident.core import ir
from trident.core.dialects import arith, torchext, tvm_ffi
from trident.input import InputTable

from .local import Local

GuardBuildFn: TypeAlias = Callable[[InputTable, ir.Context], ir.Value]
TensorMetadataFn: TypeAlias = Callable[[ir.Type, ir.Value], ir.Value]
TensorIndexedMetadataFn: TypeAlias = Callable[[ir.Type, ir.Value, ir.Value], ir.Value]
NumericOperationFn: TypeAlias = Callable[[ir.Value, ir.Value], ir.Value]


class _SkipBaseGuard(Exception): ...


class ASTVisitor(ast.NodeVisitor):
    """Compose delayed IR builders from supported Dynamo guard AST nodes."""

    _binary_ops: ClassVar[
        dict[
            type[ast.operator] | str,
            tuple[NumericOperationFn | None, NumericOperationFn | None],
        ]
    ] = {
        ast.Add: (arith.addf, arith.addi),
        ast.Sub: (arith.subf, arith.subi),
        ast.Mult: (arith.mulf, arith.muli),
        ast.Div: (arith.divf, arith.divsi),
        ast.FloorDiv: (None, arith.floordivsi),
        ast.Mod: (None, arith.remsi),
        ast.BitOr: (None, arith.ori),
        ast.RShift: (None, arith.shrsi),
        "min": (arith.minimumf, arith.minsi),
        "max": (arith.maximumf, arith.maxsi),
    }
    _compare_predicates: ClassVar[dict] = {
        ast.Eq: (arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ),
        ast.NotEq: (arith.CmpIPredicate.ne, arith.CmpFPredicate.ONE),
        ast.Lt: (arith.CmpIPredicate.slt, arith.CmpFPredicate.OLT),
        ast.LtE: (arith.CmpIPredicate.sle, arith.CmpFPredicate.OLE),
        ast.Gt: (arith.CmpIPredicate.sgt, arith.CmpFPredicate.OGT),
        ast.GtE: (arith.CmpIPredicate.sge, arith.CmpFPredicate.OGE),
    }

    @staticmethod
    def _build_binary_template(
        lhs: GuardBuildFn,
        rhs: GuardBuildFn,
        float_operation: NumericOperationFn | None,
        integer_operation: NumericOperationFn | None,
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            lhs_value = ASTVisitor._value(lhs, tree, context)
            rhs_value = ASTVisitor._value(rhs, tree, context)
            lhs_value, rhs_value, is_float = ASTVisitor._promote_numeric(
                lhs_value, rhs_value, context
            )
            operation = float_operation if is_float else integer_operation
            assert operation is not None, (
                f"unsupported numeric operation for "
                f"{'float' if is_float else 'integer'} guard operands"
            )
            return operation(lhs_value, rhs_value)

        return build

    @staticmethod
    def _build_compare_template(
        lhs: GuardBuildFn,
        rhs: GuardBuildFn,
        integer_predicate: arith.CmpIPredicate,
        float_predicate: arith.CmpFPredicate,
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            lhs_value = ASTVisitor._value(lhs, tree, context)
            rhs_value = ASTVisitor._value(rhs, tree, context)
            lhs_value, rhs_value, is_float = ASTVisitor._promote_numeric(
                lhs_value, rhs_value, context
            )
            if is_float:
                return arith.cmpf(float_predicate, lhs_value, rhs_value)
            return arith.cmpi(integer_predicate, lhs_value, rhs_value)

        return build

    def __init__(self, text: str) -> None:
        self.text = text

    @staticmethod
    def _is_integer(value: ir.Value) -> bool:
        return isinstance(value.type, ir.IntegerType)

    @staticmethod
    def _is_float(value: ir.Value) -> bool:
        return isinstance(value.type, ir.FloatType)

    @classmethod
    def _promote_numeric(
        cls,
        lhs: ir.Value,
        rhs: ir.Value,
        context: ir.Context,
    ) -> tuple[ir.Value, ir.Value, bool]:
        assert cls._is_integer(lhs) or cls._is_float(lhs)
        assert cls._is_integer(rhs) or cls._is_float(rhs)
        if cls._is_float(lhs) or cls._is_float(rhs):
            if cls._is_float(lhs) and cls._is_float(rhs):
                target = lhs.type if lhs.type.width >= rhs.type.width else rhs.type
            else:
                target = lhs.type if cls._is_float(lhs) else rhs.type
            return (
                cls._cast_numeric(lhs, target),
                cls._cast_numeric(rhs, target),
                True,
            )
        width = max(lhs.type.width, rhs.type.width)
        target = ir.IntegerType.get_signless(width, context)
        return (
            cls._cast_numeric(lhs, target),
            cls._cast_numeric(rhs, target),
            False,
        )

    @classmethod
    def _cast_numeric(
        cls,
        value: ir.Value,
        target: ir.Type,
    ) -> ir.Value:
        if value.type == target:
            return value
        if cls._is_integer(value) and cls._is_integer_value_type(target):
            if value.type.width < target.width:
                return arith.extsi(target, value)
            return arith.trunci(target, value)
        if cls._is_integer(value) and cls._is_float_type(target):
            return arith.sitofp(target, value)
        assert cls._is_float(value) and cls._is_float_type(target)
        if value.type.width < target.width:
            return arith.extf(target, value)
        return arith.truncf(target, value)

    @staticmethod
    def _is_integer_value_type(value_type: ir.Type) -> bool:
        return isinstance(value_type, ir.IntegerType)

    @staticmethod
    def _is_float_type(value_type: ir.Type) -> bool:
        return isinstance(value_type, ir.FloatType)

    def _build_comparison(
        self,
        operation: ast.cmpop,
        lhs: GuardBuildFn,
        rhs: GuardBuildFn,
    ) -> GuardBuildFn:
        predicates = self._compare_predicates.get(type(operation))
        assert predicates is not None
        integer_predicate, float_predicate = predicates
        return self._build_compare_template(
            lhs, rhs, integer_predicate, float_predicate
        )

    @staticmethod
    def _build_logical_and(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return arith.andi(lhs, rhs)

    @staticmethod
    def _build_logical_or(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return arith.ori(lhs, rhs)

    @staticmethod
    def _build_logical_not(value: ir.Value, context: ir.Context) -> ir.Value:
        i1 = ir.IntegerType.get_signless(1, context)
        true = arith.constant(i1, ir.IntegerAttr.get(i1, 1))
        return arith.xori(value, true)

    @staticmethod
    def _build_select(
        condition: ir.Value,
        true_value: ir.Value,
        false_value: ir.Value,
    ) -> ir.Value:
        return arith.select(condition, true_value, false_value)

    @staticmethod
    def _build_negate(value: ir.Value) -> ir.Value:
        zero = arith.constant(
            value.type,
            ir.FloatAttr.get(value.type, 0.0)
            if isinstance(value.type, ir.FloatType)
            else ir.IntegerAttr.get(value.type, 0),
        )
        if isinstance(value.type, ir.FloatType):
            return arith.subf(zero, value)
        assert isinstance(value.type, ir.IntegerType)
        return arith.subi(zero, value)

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
        def length(tree: InputTable, context: ir.Context) -> ir.Value:
            value = source.resolve(tree)
            assert value is not None, f"guard source cannot be resolved: {self.text!r}"
            i64 = ir.IntegerType.get_signless(64, context)
            return tvm_ffi.array_length(i64, value)

        return self._build_comparison(ast.Eq(), length, self._constant(0))

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
    def _constant_float(value: float) -> GuardBuildFn:
        return lambda _, context: arith.constant(
            (f64 := ir.F64Type.get(context)),
            ir.FloatAttr.get(f64, value),
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
            "ndimension": torchext.tensor_dim,
            "storage_offset": torchext.tensor_storage_offset,
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
        if type(node.op) not in self._binary_ops:
            return None
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if lhs is None or rhs is None:
            return None
        operations = self._binary_ops.get(type(node.op))
        assert operations is not None
        float_operation, integer_operation = operations
        return self._build_binary_template(lhs, rhs, float_operation, integer_operation)

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
            operations = self._binary_ops.get(node.func.id)
            assert operations is not None
            float_operation, integer_operation = operations
            return self._build_binary_template(
                lhs, rhs, float_operation, integer_operation
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
            torchext.tensor_dim if is_tensor_shape else tvm_ffi.array_length,
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
            i1 = ir.IntegerType.get_signless(1, context)
            true = arith.constant(i1, ir.IntegerAttr.get(i1, 1))
            return reduce(
                self._build_logical_and,
                (comparison(tree, context) for comparison in comparison_builders),
                true,
            )

        return build

    def visit_Constant(self, node: ast.Constant) -> GuardBuildFn | None:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return self._constant(node.value)
        if isinstance(node.value, float):
            return self._constant_float(node.value)
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
            "size": torchext.tensor_size,
            "stride": torchext.tensor_stride,
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
                context,
            )
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return lambda tree, context: self._build_negate(
                self._value(operand, tree, context),
            )
        return None
