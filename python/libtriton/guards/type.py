from __future__ import annotations

import re
from typing import Any, Dict, Final, Optional
from typing_extensions import override

from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import arith, tvm_ffi as tvm_ffi_d

from .check import CheckGuard


class TensorTypeGuard(CheckGuard):
    _tvm_ffi_tensor_type_index: Final[int] = 70  # kTVMFFITensor in tvm/ffi/c_api.h

    _regex_pattern: re.Pattern = re.compile(
        rf"___check_type_id\({CheckGuard._regex_variable},\s*{CheckGuard._regex_int}\),\s*type=<class 'torch\.Tensor'>"
    )

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variable: Final[str] = variable

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
    def build_ir(
        self,
        symbol_table: Dict[str, ir.Value],
        *,
        context: Optional[ir.Context] = None,
        loc: Optional[ir.Location] = None,
    ) -> ir.Operation:
        any_value: ir.Value = symbol_table[self.variable]
        i32_type: ir.Type = ir.IntegerType.get_signless(32, context=context)

        type_index: ir.Value = tvm_ffi_d.get_type_index(i32_type, any_value, loc=loc)
        expected: ir.Value = arith.constant(
            i32_type, self._tvm_ffi_tensor_type_index, loc=loc
        )
        return arith.cmpi(arith.CmpIPredicate.eq, type_index, expected, loc=loc)
