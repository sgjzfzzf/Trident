# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast

from ..codes import ExpressionCode
from ..codes.expression import ExpressionVisitor
from ..local import Local
from .base import Guard


class DuplicateInputGuard(Guard):
    """Require the two named wrapper inputs to have object identity."""

    create_fn_name = "DUPLICATE_INPUT"
    code_types = (ExpressionCode,)

    @classmethod
    def validate_code_list(cls, texts: tuple[str, ...]) -> bool:
        return len(texts) == 1

    @classmethod
    def parse_codes(
        cls,
        texts: tuple[str, ...],
        source: Local | None,
    ) -> tuple[ExpressionCode, ...] | None:
        if source is None:
            return None
        (code,) = texts
        try:
            expression = ast.parse(code, mode="eval").body
        except SyntaxError:
            return None
        operands = ExpressionVisitor.parse_identity(expression)
        if operands is None or source not in operands:
            return None
        return (ExpressionCode(code, expression),)
