# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import re
import struct
from typing import Any, Final, Optional
from typing_extensions import override

from trident._C.trident_core import ir

from .guard import Guard


class ConstantGuard(Guard):
    _regex_pattern: re.Pattern = re.compile(
        rf"{Guard._regex_variable}\s*==\s*(.+?)\s*$"
    )

    def __init__(
        self,
        variable: str,
        type_index: int,
        payload: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(variable, *args, **kwargs)
        self.type_index: Final[int] = type_index
        self.payload: Final[int] = payload

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.variable, self.type_index, self.payload))

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[ConstantGuard]:
        if not (match := cls._regex_pattern.fullmatch(code)):
            return None

        variable, expected_code = match.groups()
        expected_value: Any = ast.literal_eval(expected_code)

        # bool is a subclass of int, so check it first.
        if isinstance(expected_value, bool):
            payload = 1 if expected_value else 0
            return ConstantGuard(variable, 2, payload)
        elif isinstance(expected_value, int):
            return ConstantGuard(variable, 1, expected_value)
        elif isinstance(expected_value, float):
            payload = struct.unpack("<q", struct.pack("<d", expected_value))[0]
            return ConstantGuard(variable, 3, payload)
        else:
            return None

    @override
    def to_attribute(self, context: ir.Context) -> Optional[ir.Attribute]:
        return ir.Attribute.parse(
            f"#tvm_ffi.ConstantGuard<type_index = {self.type_index}, payload = {self.payload}>",
            context=context,
        )
