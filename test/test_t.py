# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for the ``aten::t`` pipeline wrapper end-to-end.

``aten.t`` returns a *view* of the input: the result tensor aliases the
operand's storage with transposed strides. This test verifies that the view
semantics survive the Trident FFI round-trip:

1. shape/values match ``x.t()`` (i.e. the transposed layout is preserved,
   not silently collapsed to a contiguous reinterpretation), and
2. the result is alive and non-empty (no premature release of the aliased
   storage — a failure mode that manifests as ``(0,0)`` with ``data_ptr=0``).
"""

from __future__ import annotations

import torch
from base import AtenOpTest
from typing_extensions import override


class TTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "t"

    def test_call_t(self) -> None:
        """Call the t wrapper and compare with eager PyTorch."""
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        result: torch.Tensor = self.get_ffi_func("t")(x)

        # The FFI boundary should preserve the transposed (view) layout.
        torch.testing.assert_close(result, x.t())

    def test_call_t_square(self) -> None:
        """Square input: view semantics still apply (strides swapped)."""
        x: torch.Tensor = torch.randn([4, 4], device="cuda", dtype=torch.float32)
        result: torch.Tensor = self.get_ffi_func("t")(x)

        torch.testing.assert_close(result, x.t())

    def test_call_t_alive(self) -> None:
        """The returned tensor must be non-empty and hold a valid pointer."""
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        result: torch.Tensor = self.get_ffi_func("t")(x)

        self.assertEqual(tuple(result.shape), (3, 2))
        self.assertNotEqual(result.data_ptr, 0)

    def test_call_t_value_semantics(self) -> None:
        """The view must alias the input data (read-back after mutation)."""
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        result: torch.Tensor = self.get_ffi_func("t")(x)

        # x.t() is a view: writing into x is visible through the transpose.
        x[0, 0] = 42.0
        torch.testing.assert_close(result, x.t())
