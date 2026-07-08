# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for the ``aten::empty`` op end-to-end."""

from __future__ import annotations

from typing_extensions import override

import torch
import tvm_ffi

from base import AtenOpTest


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
