# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""End-to-end tests for ``aten::clone`` storage and memory-format semantics."""

from __future__ import annotations

import torch
from base import AtenOpTest
from typing_extensions import override


class CloneTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "clone"

    @staticmethod
    def make_transposed_input() -> torch.Tensor:
        """Return a non-contiguous tensor matching the wrapper's static shape."""
        base: torch.Tensor = torch.randn([2, 32], device="cuda", dtype=torch.float32)
        result: torch.Tensor = base.t()
        if result.is_contiguous():
            raise AssertionError("test input must be non-contiguous")
        return result

    def test_clone_contiguous_materializes_storage(self) -> None:
        """Contiguous-format clone must allocate and normalize the strides."""
        x: torch.Tensor = self.make_transposed_input()
        result: torch.Tensor = self.get_ffi_func("clone")(x)

        torch.testing.assert_close(result, x)
        self.assertTrue(result.is_contiguous())
        self.assertNotEqual(result.data_ptr(), x.data_ptr())

    def test_clone_does_not_alias_input(self) -> None:
        """Mutating the input after clone must not modify the clone result."""
        x: torch.Tensor = self.make_transposed_input()
        expected: torch.Tensor = x.clone(memory_format=torch.contiguous_format)
        result: torch.Tensor = self.get_ffi_func("clone")(x)

        x[0, 0] += 1.0
        torch.testing.assert_close(result, expected)

    def test_clone_preserve_format(self) -> None:
        """Preserve-format clone must retain transposed strides without aliasing."""
        x: torch.Tensor = self.make_transposed_input()
        result: torch.Tensor = self.get_ffi_func("clone_preserve")(x)

        torch.testing.assert_close(result, x)
        self.assertEqual(result.stride(), x.stride())
        self.assertNotEqual(result.data_ptr(), x.data_ptr())
