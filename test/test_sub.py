# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for the ``aten::sub`` pipeline wrappers end-to-end."""

from __future__ import annotations

from typing_extensions import override

import torch
import tvm_ffi

from base import AtenOpTest


class SubTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "sub"

    def test_call_sub_scalar(self) -> None:
        """Call sub.Scalar wrapper and compare with eager PyTorch."""
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        result: tvm_ffi.Tensor = self.get_ffi_func("sub_scalar")(x, 0.5, 2.0)

        torch.testing.assert_close(result, x - 0.5 * 2.0)

    def test_call_sub_tensor(self) -> None:
        """Call sub.Tensor wrapper and compare with eager PyTorch."""
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        y: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        result: tvm_ffi.Tensor = self.get_ffi_func("sub_tensor")(x, y, 1.0)

        torch.testing.assert_close(result, x - y)
