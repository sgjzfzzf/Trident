# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import Any, Optional
from typing_extensions import override

from trident._C.trident_core import ir

from .guard import Guard


class DTypeGuard(Guard):
    _regex_pattern: re.Pattern = re.compile(
        rf"str\({Guard._regex_variable}\.dtype\) == 'torch\.float32'"
    )

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(variable, *args, **kwargs)

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.variable)

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[DTypeGuard]:
        if match := cls._regex_pattern.match(code):
            [variable] = match.groups()
            return DTypeGuard(variable)
        else:
            return None

    @override
    def to_attribute(self, context: ir.Context) -> Optional[ir.Attribute]:
        return ir.Attribute.parse("#tvm_ffi.DtypeGuard<>", context=context)
