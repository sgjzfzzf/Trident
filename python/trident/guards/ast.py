# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import warnings
from collections.abc import Callable
from functools import reduce
from itertools import chain, pairwise
from typing import Any, ClassVar, TypeAlias

import torch

from trident.core import ir
from trident.core.dialects import arith, torch_c, torchext
from trident.core.dialects import torch as torch_d
from trident.core.dialects.torch import (
    TorchBoolType,
    TorchFloatType,
    TorchIntType,
    TorchListType,
    TorchNonValueTensorType,
    TorchTupleType,
    TorchValueTensorType,
)
from trident.input import InputTable

from .local import Local

GuardBuildFn: TypeAlias = Callable[[InputTable, ir.Context], ir.Value]
TensorMetadataFn: TypeAlias = Callable[[ir.Value], ir.Value]
TensorIndexedMetadataFn: TypeAlias = Callable[[ir.Type, ir.Value, ir.Value], ir.Value]
TorchOperationFn: TypeAlias = Callable[[ir.Value, ir.Value], ir.Value]
TorchOperationKey: TypeAlias = type[ast.operator | ast.cmpop] | str


class _SkipGuard(Exception):
    def __init__(self, result: bool):
        super().__init__(result)
        self.result = result


class ASTVisitor(ast.NodeVisitor):
    """Compose delayed IR builders from supported Dynamo guard AST nodes."""

    _torch_ops: ClassVar[
        dict[
            type[ast.operator | ast.cmpop] | str,
            tuple[
                TorchOperationFn | None,
                TorchOperationFn | None,
            ],
        ]
    ] = {
        ast.Add: (torch_d.aten_add_int, torch_d.aten_add_float),
        ast.Sub: (torch_d.aten_sub_int, torch_d.aten_sub_float),
        ast.Mult: (torch_d.aten_mul_int, torch_d.aten_mul_float),
        ast.Div: (torch_d.aten_div_int, torch_d.aten_div_float),
        ast.FloorDiv: (torch_d.aten_floordiv_int, None),
        ast.Mod: (torch_d.aten_remainder_int, None),
        ast.BitOr: (
            lambda lhs, rhs: torch_c.from_i64(
                arith.ori(
                    torch_c.to_i64(lhs, loc=lhs.owner.location),
                    torch_c.to_i64(rhs, loc=rhs.owner.location),
                )
            ),
            None,
        ),
        ast.Eq: (torch_d.aten_eq_int, torch_d.aten_eq_float),
        ast.NotEq: (
            torch_d.aten_ne_int,
            lambda lhs, rhs: torch_d.aten___not__(torch_d.aten_eq_float(lhs, rhs)),
        ),
        ast.Lt: (torch_d.aten_lt_int, torch_d.aten_lt_float),
        ast.LtE: (
            torch_d.aten_le_int,
            lambda lhs, rhs: torch_d.aten_ge_float(rhs, lhs),
        ),
        ast.Gt: (torch_d.aten_gt_int, torch_d.aten_gt_float),
        ast.GtE: (torch_d.aten_ge_int, torch_d.aten_ge_float),
        "min": (torch_d.prim_min_int, None),
        "max": (torch_d.prim_max_int, None),
    }

    @staticmethod
    def _build_binary_template(
        lhs: GuardBuildFn,
        rhs: GuardBuildFn,
        operation_key: TorchOperationKey,
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            lhs_raw = ASTVisitor._value(lhs, tree, context)
            rhs_raw = ASTVisitor._value(rhs, tree, context)
            integer_operation, float_operation = ASTVisitor._torch_ops[operation_key]
            operation = None
            if isinstance(lhs_raw.type, TorchIntType) and isinstance(
                rhs_raw.type, TorchIntType
            ):
                operation = integer_operation
            elif isinstance(lhs_raw.type, TorchFloatType) and isinstance(
                rhs_raw.type, TorchFloatType
            ):
                operation = float_operation
            assert operation is not None, (
                "unsupported Torch numeric operation for guard operands"
            )
            return operation(lhs_raw, rhs_raw)

        return build

    @staticmethod
    def _build_compare_template(
        lhs: GuardBuildFn,
        rhs: GuardBuildFn,
        operation_key: TorchOperationKey,
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            raw_lhs = ASTVisitor._value(lhs, tree, context)
            raw_rhs = ASTVisitor._value(rhs, tree, context)
            integer_operation, float_operation = ASTVisitor._torch_ops[operation_key]
            operation = (
                integer_operation
                if isinstance(raw_lhs.type, TorchIntType)
                and isinstance(raw_rhs.type, TorchIntType)
                else float_operation
                if isinstance(raw_lhs.type, TorchFloatType)
                and isinstance(raw_rhs.type, TorchFloatType)
                else None
            )
            assert operation is not None, (
                "unsupported Torch comparison operation for guard operands"
            )
            return operation(raw_lhs, raw_rhs)

        return build

    def __init__(self, text: str) -> None:
        self.text = text

    def _build_comparison(
        self,
        operation: ast.cmpop,
        lhs: GuardBuildFn,
        rhs: GuardBuildFn,
    ) -> GuardBuildFn:
        if isinstance(operation, ast.Eq):
            integer_operation, float_operation = self._torch_ops[type(operation)]
            if integer_operation is not None:

                def build(tree: InputTable, context: ir.Context) -> ir.Value:
                    [lhs_value, rhs_value] = [
                        torchext.convert(value)
                        if isinstance(value.type, torchext.DTypeType)
                        else value
                        for value in (
                            self._value(lhs, tree, context),
                            self._value(rhs, tree, context),
                        )
                    ]
                    if isinstance(lhs_value.type, TorchIntType) and isinstance(
                        rhs_value.type, TorchIntType
                    ):
                        return integer_operation(lhs_value, rhs_value)
                    if (
                        float_operation is not None
                        and isinstance(lhs_value.type, TorchFloatType)
                        and isinstance(rhs_value.type, TorchFloatType)
                    ):
                        return float_operation(lhs_value, rhs_value)
                    equality = torchext.eq(lhs_value, rhs_value)
                    return torch_c.from_i1(equality, loc=equality.owner.location)

                return build

            def build(tree: InputTable, context: ir.Context) -> ir.Value:
                equality = torchext.eq(
                    self._value(lhs, tree, context),
                    self._value(rhs, tree, context),
                )
                return torch_c.from_i1(equality, loc=equality.owner.location)

            return build
        return self._build_compare_template(
            lhs,
            rhs,
            type(operation),
        )

    def _build_identity(self, lhs: Local, rhs: Local) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            lhs_value = lhs.resolve(tree)
            rhs_value = rhs.resolve(tree)
            assert lhs_value is not None and rhs_value is not None, (
                f"guard source cannot be resolved: {self.text!r}"
            )
            if not all(
                isinstance(
                    value.type,
                    (TorchNonValueTensorType, TorchValueTensorType),
                )
                for value in (lhs_value, rhs_value)
            ):
                equality = torchext.eq(lhs_value, rhs_value)
                return torch_c.from_i1(equality, loc=equality.owner.location)
            warnings.warn(
                "Skipping unsupported Tensor identity guard; object identity "
                f"will not be validated: {self.text!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._true(context)

        return build

    @staticmethod
    def _build_logical_and(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return torch_d.aten___and___bool(lhs, rhs)

    @staticmethod
    def _build_logical_not(value: ir.Value) -> ir.Value:
        return torch_d.aten___not__(value)

    @staticmethod
    def _build_logical_or(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        return torch_d.aten___or___bool(lhs, rhs)

    @staticmethod
    def _build_negate(value: ir.Value) -> ir.Value:
        assert isinstance(value.type, (TorchFloatType, TorchIntType))
        if isinstance(value.type, TorchFloatType):
            return torch_d.aten_neg_float(value)
        return torch_d.aten_neg_int(value)

    def _build_not_sequence(self, source: Local) -> GuardBuildFn:
        def length(tree: InputTable, context: ir.Context) -> ir.Value:
            value = source.resolve(tree)
            assert value is not None, f"guard source cannot be resolved: {self.text!r}"
            length_value = self._sequence_length(value)
            if length_value.type == ir.IntegerType.get_signless(64, context):
                return torch_c.from_i64(
                    length_value,
                    loc=length_value.owner.location,
                )
            return length_value

        constant = self._constant(0)
        assert constant is not None
        return self._build_comparison(ast.Eq(), length, constant)

    @staticmethod
    def _sequence_length(value: ir.Value) -> ir.Value | None:
        if isinstance(value.type, TorchTupleType):
            return torch_d.constant_int(len(value.type.types))
        elif isinstance(value.type, TorchListType):
            return torch_d.aten_len_t(value)

    @staticmethod
    def _build_select(
        condition: ir.Value,
        true_value: ir.Value,
        false_value: ir.Value,
    ) -> ir.Value:
        context = condition.context
        true_value, false_value = [
            (
                torch_c.to_i1(value, loc=value.owner.location)
                if isinstance(value.type, TorchBoolType)
                else torch_c.to_i64(value, loc=value.owner.location)
                if isinstance(value.type, TorchIntType)
                else torch_c.to_f64(value, loc=value.owner.location)
                if isinstance(value.type, TorchFloatType)
                else value
            )
            for value in (true_value, false_value)
        ]
        selected = arith.select(condition, true_value, false_value)
        if selected.type == ir.IntegerType.get_signless(1, context):
            return torch_c.from_i1(selected, loc=selected.owner.location)
        if selected.type == ir.IntegerType.get_signless(64, context):
            return torch_c.from_i64(selected, loc=selected.owner.location)
        if selected.type == ir.F64Type.get(context):
            return torch_c.from_f64(selected, loc=selected.owner.location)
        return selected

    def _build_tensor_indexed_metadata(
        self,
        source: Local,
        operation: TensorIndexedMetadataFn,
        index: int,
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            tensor = source.resolve(tree)
            assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
            dimension = torch_d.constant_int(index)
            if isinstance(dimension.type, TorchIntType):
                dimension = torch_c.to_i64(
                    dimension,
                    loc=dimension.owner.location,
                )
            result = operation(tensor, dimension)
            if result.type == ir.IntegerType.get_signless(64, context):
                return torch_c.from_i64(result, loc=result.owner.location)
            return result

        return build

    def _build_tensor_metadata(
        self,
        source: Local,
        operation: TensorMetadataFn,
    ) -> GuardBuildFn:
        def build(tree: InputTable, context: ir.Context) -> ir.Value:
            tensor = source.resolve(tree)
            assert tensor is not None, f"guard source cannot be resolved: {self.text!r}"
            result = operation(tensor)
            if result.type == ir.IntegerType.get_signless(64, context):
                return torch_c.from_i64(result, loc=result.owner.location)
            return result

        return build

    @staticmethod
    def _constant(value: Any) -> GuardBuildFn | None:
        if value is None:
            return lambda _, context: torch_d.constant_none()
        if isinstance(value, torch.device):
            return lambda _, context: torch_d.constant_device(
                ir.StringAttr.get(f"{value}", context=context),
            )
        if isinstance(value, torch.dtype):
            return lambda _, context: torchext.constant_dtype(
                torchext.dtype(value, context=context),
            )
        if isinstance(value, bool):
            return lambda _, context: torch_d.constant_bool(value)
        if isinstance(value, int):
            return lambda _, context: torch_d.constant_int(value)
        if isinstance(value, float):
            return lambda _, context: torch_d.constant_float(
                ir.FloatAttr.get(ir.F64Type.get(context), value),
            )
        if isinstance(value, str):
            return lambda _, context: torch_d.constant_str(
                ir.StringAttr.get(value, context),
            )
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
        return torch_d.constant_bool(False)

    @staticmethod
    def _true(context: ir.Context) -> ir.Value:
        return torch_d.constant_bool(True)

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
            "ndimension": torchext.tensor_dim,
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
        if type(node.op) not in self._torch_ops:
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
            torchext.tensor_dim if is_tensor_shape else self._sequence_length,
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
                torch_d.constant_bool(True),
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
            )
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return lambda tree, context: self._build_negate(
                self._value(operand, tree, context),
            )
        return None
