# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import Any, Final, Optional
from typing_extensions import override

import tvm_ffi

from trident._C.trident_core import ir

from .guard import Guard


class DTypeGuard(Guard):
    _regex_pattern: re.Pattern = re.compile(
        rf"str\({Guard._regex_variable}\.dtype\) == '(torch\.\w+)'"
    )

    def __init__(self, variable: str, expected: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(variable, *args, **kwargs)
        self.expected: Final[str] = expected

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.variable, self.expected))

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[DTypeGuard]:
        if match := cls._regex_pattern.match(code):
            variable, expected = match.groups()
            return DTypeGuard(variable, expected)
        else:
            return None

    @override
    def to_attribute(self, context: ir.Context) -> Optional[ir.Attribute]:
        # Strip the "torch." prefix to get a bare dtype name (e.g. "float16")
        # and delegate to tvm_ffi.dtype for code/bits/lanes resolution.
        dt: tvm_ffi.dtype = tvm_ffi.dtype(self.expected.removeprefix("torch."))
        return ir.Attribute.parse(
            f"#tvm_ffi.DtypeGuard<code = {dt.type_code}, bits = {dt.bits}, lanes = {dt.lanes}>",
            context=context,
        )
