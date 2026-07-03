# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import Any, Optional
from typing_extensions import override

from trident._C.trident_core import ir

from .guard import Guard


class TensorTypeGuard(Guard):
    _regex_pattern: re.Pattern = re.compile(
        rf"___check_type_id\({Guard._regex_variable},\s*{Guard._regex_int}\),\s*type=<class 'torch\.Tensor'>"
    )

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(variable, *args, **kwargs)

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.variable)

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[TensorTypeGuard]:
        if match := cls._regex_pattern.match(code):
            variable, _ = match.groups()
            return TensorTypeGuard(variable)
        else:
            return None

    @override
    def to_attribute(self, context: ir.Context) -> Optional[ir.Attribute]:
        return ir.Attribute.parse("#tvm_ffi.TensorTypeGuard<>", context=context)
