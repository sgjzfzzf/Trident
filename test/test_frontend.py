# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Frontend tests for container unpacking."""

from __future__ import annotations

import unittest

import torch
import trident


class FrontendTest(unittest.TestCase):
    def test_tuple_unpack(self) -> None:
        @trident.jit
        def tuple_add(x, s):
            return x + s[0] + s[1]

        x = torch.randn(4, device="cuda")
        s = (torch.randn(4, device="cuda"), torch.randn(4, device="cuda"))
        torch.testing.assert_close(tuple_add(x, s), x + s[0] + s[1])

    def test_list_unpack(self) -> None:
        @trident.jit
        def list_add(x, s):
            return x + s[0] + s[1]

        x = torch.randn(4, device="cuda")
        s = [torch.randn(4, device="cuda"), torch.randn(4, device="cuda")]
        torch.testing.assert_close(list_add(x, s), x + s[0] + s[1])

    def test_default_argument(self) -> None:
        @trident.jit
        def add_default(x, scale=2.0):
            return x * scale

        x = torch.randn(4, device="cuda")
        torch.testing.assert_close(add_default(x), x * 2.0)


if __name__ == "__main__":
    unittest.main()
