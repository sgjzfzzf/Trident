# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import re
from collections.abc import Hashable
from typing import Any, Final, Self, override

import torch
import tvm_ffi

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.input import InputTable

from ..local import Local
from .base import GuardCode


class ConstantCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        value: Any,
    ) -> None:
        super().__init__(text, source)
        self.value: Final[Any] = value

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*{source.regex}\s*(?P<operation>==|is)\s*"
            rf"(?P<literal>.+?)\s*",
            text,
        )
        if match is None:
            return None
        literal = match.group("literal")
        if match.group("operation") == "is" and literal != "None":
            return None
        if (
            dtype_match := re.fullmatch(r"torch\.([A-Za-z_][A-Za-z0-9_]*)", literal)
        ) is not None:
            (dtype,) = dtype_match.groups()
            value = getattr(torch, dtype, None)
            if not isinstance(value, torch.dtype):
                return None
        else:
            try:
                value = ast.literal_eval(literal)
            except (SyntaxError, ValueError, TypeError):
                return None
        if not isinstance(value, (type(None), bool, int, float, str, torch.dtype)):
            return None
        return cls(text, source, value)

    @property
    def key(self) -> Hashable:
        return ("constant", self.source, type(self.value), self.value)

    @override
    def build(
        self,
        tree: InputTable,
        context: ir.Context,
    ) -> ir.Value:
        source = self.source
        if source is None:
            return super().build(tree, context)
        value = self.value
        if value is None:
            expected_type = "!tvm_ffi.none"
            constant_builder = tvm_ffi_d.constant_none
            expected_attr = None
        elif isinstance(value, torch.dtype):
            expected_type = "!tvm_ffi.dtype"
            constant_builder = tvm_ffi_d.constant_dtype
            dtype = tvm_ffi.convert(value)
            expected_attr = ir.ArrayAttr.get(
                [
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(64, context), value)
                    for value in (dtype.type_code, dtype.bits, dtype.lanes)
                ]
            )
        elif isinstance(value, bool):
            expected_type = "!tvm_ffi.bool"
            constant_builder = tvm_ffi_d.constant_bool
            expected_attr = ir.BoolAttr.get(value, context=context)
        elif isinstance(value, int):
            expected_type = "!tvm_ffi.int"
            constant_builder = tvm_ffi_d.constant_int
            expected_attr = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64, context),
                value,
            )
        elif isinstance(value, float):
            expected_type = "!tvm_ffi.float"
            constant_builder = tvm_ffi_d.constant_float
            expected_attr = ir.FloatAttr.get(ir.F64Type.get(context), value)
        else:
            expected_type = "!tvm_ffi.raw_str"
            constant_builder = tvm_ffi_d.constant_raw_str
            expected_attr = ir.StringAttr.get(value, context)

        actual = source.resolve(tree)
        if actual is None:
            return super().build(tree, context)
        expected_ir_type = ir.Type.parse(expected_type, context=context)
        expected = (
            constant_builder(expected_ir_type)
            if expected_attr is None
            else constant_builder(expected_ir_type, expected_attr)
        )
        return tvm_ffi_d.eq(
            ir.IntegerType.get_signless(1, context),
            actual,
            expected,
        )
