# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from collections.abc import Hashable
from typing import Final, Self, override

from trident.core import ir
from trident.core.dialects import arith, tvm_ffi

from ..local import Local, SourceTree
from .base import GuardCode


class SequenceLengthCode(GuardCode):
    priority = 0

    def __init__(
        self,
        text: str,
        source: Local,
        expected: int,
    ) -> None:
        super().__init__(text, source)
        self.expected: Final[int] = expected

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        source_regex = source.regex
        if re.fullmatch(rf"\s*not\s+{source_regex}\s*", text) is not None:
            return cls(text, source, 0)
        elif match := re.fullmatch(
            rf"\s*len\(\s*{source_regex}\s*\)\s*==\s*(?P<length>\d+)\s*",
            text,
        ):
            return (
                cls(text, source, int(match.group("length")))
                if match is not None
                else None
            )
        else:
            return None

    @property
    def depth(self) -> int:
        assert self.source is not None
        return self.source.depth

    @property
    def key(self) -> Hashable:
        return ("sequence-length", self.source, self.expected)

    @override
    def build(
        self,
        tree: SourceTree,
        context: ir.Context,
    ) -> ir.Value:
        source = self.source
        if source is None:
            return super().build(tree, context)
        value = source.resolve(tree)
        if value is None:
            return super().build(tree, context)
        i64 = ir.IntegerType.get_signless(64, context)
        actual = tvm_ffi.array_length(i64, value)
        return arith.cmpi(
            arith.CmpIPredicate.eq,
            actual,
            arith.constant(
                i64,
                ir.IntegerAttr.get(i64, self.expected),
            ),
        )
