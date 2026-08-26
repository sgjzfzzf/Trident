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

from ..local import Local, SourceTree
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
        tree: SourceTree,
        context: ir.Context,
    ) -> ir.Value:
        source = self.source
        if source is None:
            return super().build(tree, context)
        if self.value is None:
            expected_type = "!tvm_ffi.none"
            expected_attr: ir.Attribute = ir.UnitAttr.get(context)
        elif isinstance(self.value, torch.dtype):
            expected_type = "!tvm_ffi.dtype"
            dtype = tvm_ffi.convert(self.value)
            expected_attr = ir.ArrayAttr.get(
                [
                    ir.IntegerAttr.get(ir.IntegerType.get_signless(64, context), value)
                    for value in (dtype.type_code, dtype.bits, dtype.lanes)
                ]
            )
        elif isinstance(self.value, bool):
            expected_type = "!tvm_ffi.bool"
            expected_attr = ir.BoolAttr.get(self.value, context=context)
        elif isinstance(self.value, int):
            expected_type = "!tvm_ffi.int"
            expected_attr = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64, context),
                self.value,
            )
        elif isinstance(self.value, float):
            expected_type = "!tvm_ffi.float"
            expected_attr = ir.FloatAttr.get(
                ir.F64Type.get(context),
                self.value,
            )
        elif isinstance(self.value, str):
            expected_type = "!tvm_ffi.string"
            expected_attr = ir.StringAttr.get(self.value, context)

        actual = source.resolve(tree)
        if actual is None:
            return super().build(tree, context)
        expected = tvm_ffi_d.constant(
            ir.Type.parse(expected_type, context=context),
            expected_attr,
        )
        return tvm_ffi_d.eq(
            ir.IntegerType.get_signless(1, context),
            actual,
            expected,
        )
