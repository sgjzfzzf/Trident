# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from collections.abc import Hashable
from typing import Final, Self, override

from trident.core import ir
from trident.input import InputTable

from ..ast import ASTVisitor
from ..local import Local
from .base import GuardBuilder, GuardBuildFn, GuardCode


class ConstantCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        expression: ast.expr,
    ) -> None:
        super().__init__(text, source)
        self.expression: Final[ast.expr] = expression
        self.build_fn: Final[GuardBuildFn | None] = ASTVisitor(text).visit(expression)

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        try:
            expression = ast.parse(text.strip(), mode="eval").body
        except SyntaxError:
            return None
        if (
            not isinstance(expression, ast.Compare)
            or len(expression.ops) != 1
            or len(expression.comparators) != 1
            or Local.from_expression(expression.left) != source
        ):
            return None
        [operation] = expression.ops
        [comparator] = expression.comparators
        if (
            isinstance(operation, ast.Is)
            and isinstance(comparator, ast.Constant)
            and comparator.value is None
        ) or (
            isinstance(operation, ast.Eq)
            and Local.from_expression(comparator) is None
            and ASTVisitor(text).visit(comparator) is not None
        ):
            return cls(text, source, expression)
        else:
            return None

    @property
    def key(self) -> Hashable:
        return (
            "constant",
            self.source,
            ast.dump(self.expression, include_attributes=False),
        )

    @override
    def build(
        self,
        tree: InputTable,
        context: ir.Context,
    ) -> ir.Value:
        if self.build_fn is None:
            return super().build(tree, context)
        return self.build_fn(tree, context)

    def to_builder(self) -> GuardBuilder:
        assert self.build_fn is not None, (
            f"guard expression cannot be lowered: {self.text!r}"
        )
        return GuardBuilder(code=self, build_fn=self.build_fn)
