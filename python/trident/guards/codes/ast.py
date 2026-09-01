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


class ASTCode(GuardCode):
    priority = 2

    def __init__(
        self,
        text: str,
        expression: ast.expr,
    ) -> None:
        super().__init__(text, None)
        self.expression: Final[ast.expr] = expression
        self.build_fn: Final[GuardBuildFn | None] = ASTVisitor(text).build(expression)

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
        code = cls(text, expression)
        return code if code.build_fn is not None else None

    @property
    def key(self) -> Hashable:
        return ("ast", ast.dump(self.expression, include_attributes=False))

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
