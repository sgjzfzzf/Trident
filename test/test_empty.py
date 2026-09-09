# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for the ``aten::empty`` op end-to-end."""

from __future__ import annotations

import torch
import tvm_ffi
from typing_extensions import override

from test.base import AtenOpTest


class EmptyTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "empty"

    def test_call_empty(self) -> None:
        """Call aten::empty with shape [3, 4], device=cuda, dtype=float32."""
        result: tvm_ffi.Tensor = self.get_ffi_func("empty")(
            [3, 4], tvm_ffi.device("cuda"), 6
        )
        self.assertEqual(result.shape, torch.Size([3, 4]))
        self.assertEqual(result.dtype, tvm_ffi._dtype.float32)

    def test_call_empty_like(self) -> None:
        """Call empty_like and verify output shape/dtype match input."""
        x: torch.Tensor = torch.randn([200, 200, 26], device="cuda")
        result: torch.Tensor = self.get_ffi_func("empty_like")(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
