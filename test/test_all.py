# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""End-to-end tests for list-valued reduction dimensions."""

from __future__ import annotations

import unittest

import torch
import trident
from base import TridentTestCase


@trident.jit
def all_over_list_dims(x: torch.Tensor) -> torch.Tensor:
    return torch.all(x, dim=[1, 0], keepdim=False)


class AllListDimsTest(TridentTestCase):
    def test_all_with_list_dims(self) -> None:
        x = torch.ones((7, 4, 11, 1), dtype=torch.float32)
        result = all_over_list_dims(x)

        expected = torch.all(x, dim=[1, 0], keepdim=False)
        torch.testing.assert_close(result, expected)
        self.assertEqual(result.shape, torch.Size((11, 1)))
        self.assertEqual(result.dtype, torch.bool)
