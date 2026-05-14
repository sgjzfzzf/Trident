from __future__ import annotations

import re
from typing import Any, Dict, Final, Optional
from typing_extensions import override

from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import arith, dlpack, tvm_ffi as tvm_ffi_d

from .check import CheckGuard


class StorageOffsetGuard(CheckGuard):
    _regex_pattern: re.Pattern = re.compile(
        rf"{CheckGuard._regex_variable}\.storage_offset\(\) == {CheckGuard._regex_int}"
    )

    def __init__(self, variable: str, expected: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variable: Final[str] = variable
        self.expected: Final[int] = expected

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.variable, self.expected))

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[StorageOffsetGuard]:
        if match := cls._regex_pattern.match(code):
            variable, expected = match.groups()
            return StorageOffsetGuard(variable, int(expected))
        else:
            return None

    @override
    def build_ir(
        self,
        symbol_table: Dict[str, ir.Value],
        *,
        context: Optional[ir.Context] = None,
        loc: Optional[ir.Location] = None,
    ) -> ir.Operation:
        any_value: ir.Value = symbol_table[self.variable]
        dl_tensor_type: ir.Type = ir.Type.parse("!dlpack.tensor", context=context)
        i64_type: ir.Type = ir.IntegerType.get_signless(64, context=context)

        dl_tensor: ir.Value = tvm_ffi_d.to(dl_tensor_type, any_value, loc=loc)
        byte_offset: ir.Value = dlpack.byte_offset(i64_type, dl_tensor, loc=loc)
        expected: ir.Value = arith.constant(i64_type, self.expected, loc=loc)
        return arith.cmpi(arith.CmpIPredicate.eq, byte_offset, expected, loc=loc)
