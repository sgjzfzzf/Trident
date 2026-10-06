# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Cached Triton launch through the direct semantic importer."""

from __future__ import annotations

import unittest

import torch
import trident
import triton
import triton.language as tl

from test.base import TridentTestCase


@triton.jit
def _add_kernel(x, y, output, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    tl.store(
        output + offsets, tl.load(x + offsets, mask) + tl.load(y + offsets, mask), mask
    )


class TritonTest(TridentTestCase):
    def test_cached_dynamic_launch(self) -> None:
        @trident.jit(dynamic=True)
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            size = x.numel()
            _add_kernel[lambda meta: (triton.cdiv(size, meta["BLOCK"]), 1, 1)](
                x, y, output, size, 128
            )
            return output

        for size in (257, 513, 777):
            x = torch.randn(size, device="cuda")
            y = torch.randn_like(x)
            add(x, y)
            torch.testing.assert_close(add(x, y), x + y)
        self.assertEqual(len(add._sub_modules), 1)
        text = str(add._sub_modules[0])
        self.assertIn("torchext.trident_kernel_launch", text)
        self.assertIn("tvm_ffi.tensor.size", text)
        self.assertNotIn("!torch.", text)


if __name__ == "__main__":
    unittest.main()
