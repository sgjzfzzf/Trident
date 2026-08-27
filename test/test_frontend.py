# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Frontend compilation tests."""

from __future__ import annotations

import unittest
from collections.abc import Sequence

import torch
import trident
from base import TridentTestCase


class FrontendTest(TridentTestCase):
    def test_local_mutation(self) -> None:
        @trident.jit
        def zero_empty_like(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x).zero_()

        x = torch.randn(4, device="cuda")
        torch.testing.assert_close(zero_empty_like(x), torch.zeros_like(x))

    def test_tuple_unpack(self) -> None:
        @trident.jit
        def tuple_add(
            x: torch.Tensor, s: tuple[torch.Tensor, torch.Tensor]
        ) -> torch.Tensor:
            return x + s[0] + s[1]

        x = torch.randn(4, device="cuda")
        s = (torch.randn(4, device="cuda"), torch.randn(4, device="cuda"))
        torch.testing.assert_close(tuple_add(x, s), x + s[0] + s[1])

    def test_list_unpack(self) -> None:
        @trident.jit
        def list_add(x: torch.Tensor, s: Sequence[torch.Tensor]) -> torch.Tensor:
            return x + s[0] + s[1]

        x = torch.randn(4, device="cuda")
        s = [torch.randn(4, device="cuda"), torch.randn(4, device="cuda")]
        torch.testing.assert_close(list_add(x, s), x + s[0] + s[1])


if __name__ == "__main__":
    unittest.main()
