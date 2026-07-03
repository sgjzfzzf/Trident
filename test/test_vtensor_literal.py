# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for the ``torch.vtensor.literal`` pipeline wrapper end-to-end."""

from __future__ import annotations

from typing_extensions import override

import torch
import tvm_ffi

from base import AtenOpTest


class VTensorLiteralTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "vtensor_literal"

    def test_call_vtensor_literal_splat(self) -> None:
        """Call vtensor_literal and verify it returns a TVM FFI tensor."""
        result: tvm_ffi.Tensor = self.get_ffi_func("vtensor_literal_splat")()
        self.assertIsInstance(result, tvm_ffi.Tensor)

    def test_call_vtensor_literal_nonsplat(self) -> None:
        """Call nonsplat wrapper and verify multi-item tensor metadata."""
        result: tvm_ffi.Tensor = self.get_ffi_func("vtensor_literal_nonsplat")()

        self.assertEqual(result.shape, torch.Size([2, 3]))
        self.assertEqual(result.dtype, tvm_ffi._dtype.float32)
