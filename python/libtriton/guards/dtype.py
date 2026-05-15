from __future__ import annotations

import re
from typing import Any, Dict, Final, Optional
from typing_extensions import override

from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import arith, dlpack, tvm_ffi as tvm_ffi_d

from .check import CheckGuard


class DTypeGuard(CheckGuard):
    _dl_dtype_float: Final[int] = 2
    _dl_dtype_bits_float32: Final[int] = 32
    _dl_dtype_lanes_default: Final[int] = 1
    _regex_pattern: re.Pattern = re.compile(
        rf"str\({CheckGuard._regex_variable}\.dtype\) == 'torch\.float32'"
    )

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variable: Final[str] = variable

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
    def build_ir(
        self,
        symbol_table: Dict[str, ir.Value],
        *,
        context: Optional[ir.Context] = None,
        loc: Optional[ir.Location] = None,
    ) -> ir.Operation:
        i8_type: ir.Type = ir.IntegerType.get_signless(8, context=context)
        i16_type: ir.Type = ir.IntegerType.get_signless(16, context=context)

        dl_datatype: ir.Value = dlpack.dtype(
            ir.Type.parse("!dlpack.datatype", context=context),
            tvm_ffi_d.to(
                ir.Type.parse("!dlpack.tensor", context=context),
                symbol_table[self.variable],
                loc=loc,
            ),
            loc=loc,
        )

        return arith.andi(
            arith.andi(
                arith.cmpi(
                    arith.CmpIPredicate.eq,
                    dlpack.dtype_code(i8_type, dl_datatype, loc=loc),
                    arith.constant(i8_type, self._dl_dtype_float, loc=loc),
                    loc=loc,
                ),
                arith.cmpi(
                    arith.CmpIPredicate.eq,
                    dlpack.dtype_bits(i8_type, dl_datatype, loc=loc),
                    arith.constant(i8_type, self._dl_dtype_bits_float32, loc=loc),
                    loc=loc,
                ),
                loc=loc,
            ),
            arith.cmpi(
                arith.CmpIPredicate.eq,
                dlpack.dtype_lanes(i16_type, dl_datatype, loc=loc),
                arith.constant(i16_type, self._dl_dtype_lanes_default, loc=loc),
                loc=loc,
            ),
            loc=loc,
        )
