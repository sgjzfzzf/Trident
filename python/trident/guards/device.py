# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
import torch
from typing import Any, Final, Optional
from typing_extensions import override

from trident._C.trident_core import ir

from .guard import Guard


class CUDADeviceGuard(Guard):
    _regex_pattern: re.Pattern = re.compile(
        rf"str\({Guard._regex_variable}\.device\) == '(cuda:\d+)'"
    )

    def __init__(self, variable: str, expected: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(variable, *args, **kwargs)
        self.expected: Final[str] = expected

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.variable, self.expected))

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[CUDADeviceGuard]:
        if match := cls._regex_pattern.match(code):
            variable, expected = match.groups()
            return CUDADeviceGuard(variable, expected)
        else:
            return None

    @override
    def to_attribute(self, context: ir.Context) -> Optional[ir.Attribute]:
        return ir.Attribute.parse(
            f"#tvm_ffi.CudaDeviceGuard<device_type = 2, device_index = {torch.device(self.expected).index}>",
            context=context,
        )
