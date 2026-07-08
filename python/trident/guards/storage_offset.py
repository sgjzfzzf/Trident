# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import re
from typing import Any, Final, Optional
from typing_extensions import override

from trident._C.trident_core import ir

from .guard import Guard


class StorageOffsetGuard(Guard):
    _regex_pattern: re.Pattern = re.compile(
        rf"{Guard._regex_variable}\.storage_offset\(\) == {Guard._regex_int}"
    )

    def __init__(self, variable: str, expected: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(variable, *args, **kwargs)
        self.expected: Final[int] = expected

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.variable, self.expected))

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[StorageOffsetGuard]:
        if match := cls._regex_pattern.match(code):
            variable, expected = match.groups()
            return StorageOffsetGuard(variable, ast.literal_eval(expected))
        else:
            return None

    @override
    def to_attribute(self, context: ir.Context) -> Optional[ir.Attribute]:
        return ir.Attribute.parse(
            f"#tvm_ffi.StorageOffsetGuard<expected = {self.expected}>",
            context=context,
        )
