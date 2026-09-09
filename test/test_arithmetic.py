# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""End-to-end tests for scalar and tensor arithmetic wrappers."""

from __future__ import annotations

import torch
import tvm_ffi
from typing_extensions import override

from test.base import AtenOpTest


class ArithmeticTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "arithmetic"

    def test_scalar_arithmetic(self) -> None:
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        cases = (
            ("mul_scalar", (x, 0.5), x * 0.5),
            ("sub_scalar", (x, 0.5, 2.0), x - 0.5 * 2.0),
        )
        for name, arguments, expected in cases:
            with self.subTest(operation=name):
                result: tvm_ffi.Tensor = self.get_ffi_func(name)(*arguments)
                torch.testing.assert_close(result, expected)

    def test_tensor_sub(self) -> None:
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        y: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        result: tvm_ffi.Tensor = self.get_ffi_func("sub_tensor")(x, y, 1.0)

        torch.testing.assert_close(result, x - y)
