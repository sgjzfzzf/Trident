# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import Self

from trident.core import ir
from trident.core.dialects import arith
from trident.input import InputTable

from ..local import Local
from .base import GuardCode


class TypeIdCode(GuardCode):
    """A type-id template covered by the wrapper function signature."""

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        source_regex = source.regex
        match = re.fullmatch(
            rf"\s*___check_type_id\(\s*{source_regex}\s*,\s*\d+\s*\)\s*,\s*"
            rf"type\s*=\s*<class\s+(?P<quote>['\"])"
            rf"(?P<type_name>[A-Za-z_][A-Za-z0-9_.]*)"
            rf"(?P=quote)>\s*",
            text,
        )
        return cls(text, source) if match is not None else None

    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        """The ABI type tag and Torch signature cover this guard."""
        i1 = ir.IntegerType.get_signless(1, context)
        return arith.constant(i1, ir.IntegerAttr.get(i1, 1))
