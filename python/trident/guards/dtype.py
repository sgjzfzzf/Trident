# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import Any, Dict, Final, Optional, Tuple
from typing_extensions import override

from trident._C.trident_core import ir

from .guard import Guard


class DTypeGuard(Guard):
    _regex_pattern: re.Pattern = re.compile(
        rf"str\({Guard._regex_variable}\.dtype\) == '(torch\.float16|torch\.float32)'"
    )

    _dtype_to_dl: Final[Dict[str, Tuple[int, int, int]]] = {
        "torch.float16": (2, 16, 1),
        "torch.float32": (2, 32, 1),
    }

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
        if self.expected not in self._dtype_to_dl:
            code, bits, lanes = self._dtype_to_dl[self.expected]
            return ir.Attribute.parse(
                f"#tvm_ffi.DtypeGuard<code = {code}, bits = {bits}, lanes = {lanes}>",
                context=context,
            )
        else:
            return None
