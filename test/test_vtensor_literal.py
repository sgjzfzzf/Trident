# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for the ``torch.vtensor.literal`` pipeline wrapper end-to-end."""

from __future__ import annotations

import torch
import tvm_ffi
from typing_extensions import override

from test.base import AtenOpTest


class VTensorLiteralTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "value-tensor-literal"

    def test_call_vtensor_literals(self) -> None:
        """The splat and nonsplat wrappers preserve shape, dtype, and values."""
        cases = (
            (
                "nonsplat",
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    device="cuda",
                    dtype=torch.float32,
                ),
            ),
            (
                "splat",
                torch.full([2, 3], 1.25, device="cuda", dtype=torch.float32),
            ),
        )
        for name, expected in cases:
            with self.subTest(literal=name):
                result: tvm_ffi.Tensor = self.get_ffi_func(f"vtensor_literal_{name}")()
                torch.testing.assert_close(torch.from_dlpack(result), expected)
