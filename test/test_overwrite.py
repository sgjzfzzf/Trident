# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""End-to-end tests for in-place tensor overwrite lowering."""

from __future__ import annotations

import torch
from base import AtenOpTest
from typing_extensions import override


class OverwriteTest(AtenOpTest):
    @classmethod
    @override
    def op_name(cls) -> str:
        return "overwrite"

    def test_copy_mutation(self) -> None:
        """The copy wrapper writes source contents into the destination."""
        source: torch.Tensor = torch.randn([2, 3], device="cuda")
        destination: torch.Tensor = torch.zeros_like(source)

        self.get_ffi_func("overwrite")(source, destination)

        torch.testing.assert_close(destination, source)
