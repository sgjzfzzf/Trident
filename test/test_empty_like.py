# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for the ``aten::empty_like`` op end-to-end."""

from __future__ import annotations

from typing_extensions import override

import torch

from base import AtenOpTest


class EmptyLikeTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "empty_like"

    def test_call_empty_like(self) -> None:
        """Call empty_like and verify output shape/dtype match input."""
        x: torch.Tensor = torch.randn([200, 200, 26], device="cuda")
        result: torch.Tensor = self.get_ffi_func("empty_like")(x)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.dtype, x.dtype)
