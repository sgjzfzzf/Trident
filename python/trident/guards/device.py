# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import Any, Final, Optional
from typing_extensions import override

import tvm_ffi

from trident.core import ir

from .guard import Guard


class CUDADeviceGuard(Guard):
    _regex_pattern: re.Pattern = re.compile(
        rf"str\({Guard._regex_variable}\.device\) == '(\w+(?::\d+)?)'"
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
        dev: tvm_ffi.Device = tvm_ffi.device(self.expected)
        return ir.Attribute.parse(
            f"#tvm_ffi.CudaDeviceGuard<device_type = {dev.dlpack_device_type()}, device_index = {dev.index}>",
            context=context,
        )
