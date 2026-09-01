# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..codes import ASTCode, GuardBuilder, TypeIdCode
from ..local import Local
from .base import Guard

if TYPE_CHECKING:
    import torch._guards


class SequenceLengthGuard(Guard):
    create_fn_name = "SEQUENCE_LENGTH"
    code_types = (TypeIdCode, ASTCode)

    @classmethod
    def parse_source(cls, guard: torch._guards.Guard) -> Local | None:
        try:
            expression = ast.parse(guard.name, mode="eval").body
        except SyntaxError:
            return None
        if isinstance(expression, ast.Attribute) and expression.attr == "shape":
            return Local.from_expression(expression.value)
        return Local.from_expression(expression)

    @classmethod
    def parse_codes(
        cls,
        texts: list[str],
        source: Local | None,
    ) -> list[GuardBuilder] | None:
        expressions = [ASTCode.parse(text, source) for text in texts]
        has_tensor_shape = any(
            expression is not None
            and any(
                isinstance(node, ast.Attribute) and node.attr == "shape"
                for node in ast.walk(expression.expression)
            )
            for expression in expressions
        )
        parsed_codes: list[GuardBuilder] = []
        for text, expression in zip(texts, expressions):
            if expression is not None:
                parsed_codes.append(expression.to_builder())
                continue
            if has_tensor_shape:
                continue
            type_code = TypeIdCode.parse(text, source)
            if type_code is None:
                return None
            parsed_codes.append(type_code.to_builder())
        return parsed_codes
