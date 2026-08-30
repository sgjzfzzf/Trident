# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from typing import TypeAlias

from trident.core import ir
from trident.input import InputTable

Subscript: TypeAlias = int | str


class Local:
    """A local ABI argument followed by zero or more subscripts."""

    __slots__ = ("_subscripts",)

    def __init__(self, subscripts: Sequence[Subscript]) -> None:
        assert subscripts, "local subscripts must not be empty"
        root, *follows = subscripts
        assert isinstance(root, str) and root, (
            "local subscripts must start with a non-empty string"
        )
        assert all(
            isinstance(step, (int, str)) and not isinstance(step, bool)
            for step in follows
        ), "local subscripts must be integers or strings"
        self._subscripts: tuple[Subscript, ...] = tuple(subscripts)

    @property
    def subscripts(self) -> tuple[Subscript, ...]:
        return self._subscripts

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Local) and self._subscripts == other._subscripts

    def __hash__(self) -> int:
        return hash(self._subscripts)

    def __repr__(self) -> str:
        return f"Local(subscripts={self._subscripts!r})"

    @classmethod
    def parse(cls, text: str) -> Local | None:
        try:
            expression = ast.parse(text, mode="eval").body
        except SyntaxError:
            return None
        return cls.from_expression(expression)

    @classmethod
    def from_expression(cls, expression: ast.expr) -> Local | None:
        def parse(node: ast.expr) -> list[Subscript] | None:
            if not isinstance(node, ast.Subscript):
                return None

            if isinstance(node.value, ast.Name) and node.value.id == "L":
                if isinstance(node.slice, ast.Constant) and isinstance(
                    node.slice.value, str
                ):
                    return [node.slice.value]
                return None

            if not (
                isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, (int, str))
                and not isinstance(node.slice.value, bool)
            ):
                return None

            subscripts = parse(node.value)
            if subscripts is None:
                return None
            return [*subscripts, node.slice.value]

        subscripts = parse(expression)
        return cls(subscripts) if subscripts is not None else None

    @property
    def depth(self) -> int:
        return len(self._subscripts) - 1

    @property
    def literal(self) -> str:
        return f"L{''.join(f'[{subscript!r}]' for subscript in self._subscripts)}"

    @property
    def regex(self) -> str:
        return re.escape(self.literal)

    def resolve(self, tree: InputTable) -> ir.Value | None:
        return tree[self._subscripts]
