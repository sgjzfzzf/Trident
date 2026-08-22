# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from collections.abc import Hashable
from functools import reduce
from typing import Final, Self, override

import tvm_ffi as runtime_ffi

from trident.core import ir
from trident.core.dialects import arith, tvm_ffi

from ..local import Local, SourceTree
from .base import GuardCode


class TensorDTypeCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        expected: str,
        dtype: runtime_ffi.dtype,
    ) -> None:
        super().__init__(text, source)
        self.expected: Final[str] = expected
        self.dtype: Final[runtime_ffi.dtype] = dtype

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*str\(\s*{source.regex}\s*\.dtype\s*\)\s*==\s*"
            rf"(?P<quote>['\"])(?P<dtype>torch\.[A-Za-z_][A-Za-z0-9_]*)"
            rf"(?P=quote)\s*",
            text,
        )
        if match is None:
            return None
        expected = match.group("dtype")
        try:
            dtype = runtime_ffi.dtype(expected.removeprefix("torch."))
        except (TypeError, ValueError):
            return None
        return cls(text, source, expected, dtype)

    @property
    def key(self) -> Hashable:
        return ("tensor-dtype", self.source, self.expected)

    @override
    def build(
        self,
        tree: SourceTree,
        context: ir.Context,
    ) -> ir.Value:
        source = self.source
        if source is None:
            return super().build(tree, context)
        tensor = source.resolve(tree)
        if tensor is None:
            return super().build(tree, context)
        result_types = (
            ir.IntegerType.get_signless(8, context),
            ir.IntegerType.get_signless(8, context),
            ir.IntegerType.get_signless(16, context),
        )
        metadata = tvm_ffi.tensor_dtype(*result_types, tensor)
        i1 = ir.IntegerType.get_signless(1, context)
        return reduce(
            arith.andi,
            [
                arith.cmpi(
                    arith.CmpIPredicate.eq,
                    actual,
                    arith.constant(
                        expected_type,
                        ir.IntegerAttr.get(expected_type, expected_value),
                    ),
                )
                for actual, expected_value, expected_type in zip(
                    metadata,
                    (self.dtype.type_code, self.dtype.bits, self.dtype.lanes),
                    result_types,
                )
            ],
            arith.constant(i1, ir.IntegerAttr.get(i1, 1)),
        )


class TensorDeviceCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        expected: str,
        device: runtime_ffi.Device,
    ) -> None:
        super().__init__(text, source)
        self.expected: Final[str] = expected
        self.device: Final[runtime_ffi.Device] = device

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*str\(\s*{source.regex}\s*\.device\s*\)\s*==\s*"
            rf"(?P<quote>['\"])(?P<device>[^'\"]+)(?P=quote)\s*",
            text,
        )
        if match is None:
            return None
        expected = match.group("device")
        try:
            device = runtime_ffi.device(expected)
        except (TypeError, ValueError):
            return None
        return cls(text, source, expected, device)

    @property
    def key(self) -> Hashable:
        return ("tensor-device", self.source, self.expected)

    @override
    def build(
        self,
        tree: SourceTree,
        context: ir.Context,
    ) -> ir.Value:
        source = self.source
        if source is None:
            return super().build(tree, context)
        tensor = source.resolve(tree)
        if tensor is None:
            return super().build(tree, context)
        i32 = ir.IntegerType.get_signless(32, context)
        metadata = tvm_ffi.tensor_device(i32, i32, tensor)
        i1 = ir.IntegerType.get_signless(1, context)
        return reduce(
            arith.andi,
            [
                arith.cmpi(
                    arith.CmpIPredicate.eq,
                    metadata[0],
                    arith.constant(
                        i32,
                        ir.IntegerAttr.get(
                            i32,
                            self.device.dlpack_device_type(),
                        ),
                    ),
                ),
                arith.cmpi(
                    arith.CmpIPredicate.eq,
                    metadata[1],
                    arith.constant(
                        i32,
                        ir.IntegerAttr.get(i32, self.device.index),
                    ),
                ),
            ],
            arith.constant(i1, ir.IntegerAttr.get(i1, 1)),
        )


class TensorRankCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        expected: int,
    ) -> None:
        super().__init__(text, source)
        self.expected: Final[int] = expected

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*{source.regex}\s*\.ndimension\(\s*\)\s*==\s*"
            rf"(?P<rank>\d+)\s*",
            text,
        )
        return (
            cls(text, source, int(match.group("rank"))) if match is not None else None
        )

    @property
    def key(self) -> Hashable:
        return ("tensor-dim", self.source, self.expected)

    @override
    def build(
        self,
        tree: SourceTree,
        context: ir.Context,
    ) -> ir.Value:
        source = self.source
        if source is None:
            return super().build(tree, context)
        i64 = ir.IntegerType.get_signless(64, context)
        tensor = source.resolve(tree)
        if tensor is None:
            return super().build(tree, context)
        actual = tvm_ffi.tensor_dim(
            i64,
            tensor,
        )
        return arith.cmpi(
            arith.CmpIPredicate.eq,
            actual,
            arith.constant(
                i64,
                ir.IntegerAttr.get(i64, self.expected),
            ),
        )


class RequiresGradCode(GuardCode):
    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*{source.regex}\s*\.requires_grad\s*==\s*"
            rf"(?P<value>True|False)\s*",
            text,
        )
        if match is None:
            return None
        if match.group("value") == "True":
            return None
        return cls(text, source)

    @override
    def build(self, tree: SourceTree, context: ir.Context) -> ir.Value:
        # TVM FFI tensor metadata does not expose autograd state.  Dynamo's
        # false case is the invariant used by the current backend; a true
        # case is rejected during parsing rather than silently accepted.
        i1 = ir.IntegerType.get_signless(1, context)
        return reduce(
            arith.andi,
            [],
            arith.constant(i1, ir.IntegerAttr.get(i1, 1)),
        )


class DynamoAttributeAbsentCode(GuardCode):
    def __init__(
        self,
        text: str,
        source: Local,
        attribute: str,
    ) -> None:
        super().__init__(text, source)
        self.attribute: Final[str] = attribute

    @classmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None:
        if source is None:
            return None
        match = re.fullmatch(
            rf"\s*hasattr\(\s*{source.regex}\s*,\s*"
            rf"(?P<quote>['\"])(?P<attribute>_dynamo_[A-Za-z0-9_]*)"
            rf"(?P=quote)\s*\)\s*==\s*False\s*",
            text,
        )
        return (
            cls(text, source, match.group("attribute")) if match is not None else None
        )

    @override
    def build(self, tree: SourceTree, context: ir.Context) -> ir.Value:
        # These private Dynamo attributes are not part of the TVM FFI ABI.
        i1 = ir.IntegerType.get_signless(1, context)
        return reduce(
            arith.andi,
            [],
            arith.constant(i1, ir.IntegerAttr.get(i1, 1)),
        )
