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
from typing_extensions import override

from test.base import AtenOpTest


class TTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "transpose"

    def test_call_t_rectangular_view(self) -> None:
        """The transpose has the expected layout and aliases its input."""
        x: torch.Tensor = torch.randn([2, 3], device="cuda", dtype=torch.float32)
        result: torch.Tensor = self.get_ffi_func("t")(x)

        torch.testing.assert_close(result, x.t())
        self.assertEqual(result.shape, torch.Size([3, 2]))
        self.assertEqual(result.stride(), x.t().stride())
        self.assertNotEqual(result.data_ptr(), 0)

        x[0, 0] = 42.0
        self.assertEqual(result[0, 0].item(), 42.0)
