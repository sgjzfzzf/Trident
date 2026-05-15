from __future__ import annotations

import re
from typing import Any, Dict, Final, Optional
from typing_extensions import override

from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import arith, dlpack, tvm_ffi as tvm_ffi_d

from .check import CheckGuard


class DimensionGuard(CheckGuard):
    _regex_pattern: re.Pattern = re.compile(
        rf"{CheckGuard._regex_variable}\.ndimension\(\) == {CheckGuard._regex_int}"
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
    def _parse(cls, code: str) -> Optional[DimensionGuard]:
        if match := cls._regex_pattern.match(code):
            variable, expected = match.groups()
            return DimensionGuard(variable, int(expected))
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
        i32_type: ir.Type = ir.IntegerType.get_signless(32, context=context)

        return arith.cmpi(
            arith.CmpIPredicate.eq,
            dlpack.ndim(
                i32_type,
                tvm_ffi_d.to(
                    ir.Type.parse("!dlpack.tensor", context=context),
                    symbol_table[self.variable],
                    loc=loc,
                ),
                loc=loc,
            ),
            arith.constant(i32_type, self.expected, loc=loc),
            loc=loc,
        )
