# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for the ``aten::mul.Scalar`` pipeline wrapper end-to-end."""

from __future__ import annotations

from typing_extensions import override

import torch

from base import AtenOpTest


class MulScalarTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "mul_scalar"

    def test_call_mul_scalar(self) -> None:
        """Call mul.Scalar and verify the output matches eager PyTorch."""
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        result: torch.Tensor = self.get_ffi_func("mul_scalar")(x, 0.5)

        torch.testing.assert_close(result, x * 0.5)
