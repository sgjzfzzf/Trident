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
    def test_device_argument(self) -> None:
        @trident.jit
        def empty_on_device(
            x: torch.Tensor,
            device: torch.device,
        ) -> torch.Tensor:
            return torch.empty_like(x, device=device).zero_()

        x = torch.randn(4, device="cuda")
        device = torch.device("cuda")
        torch.testing.assert_close(empty_on_device(x, device), torch.zeros_like(x))
        torch.testing.assert_close(empty_on_device(x, device), torch.zeros_like(x))
        cpu_result = empty_on_device(x, torch.device("cpu"))
        self.assertEqual(cpu_result.device, torch.device("cpu"))
        torch.testing.assert_close(cpu_result, torch.zeros_like(cpu_result))

    def test_dtype_argument(self) -> None:
        @trident.jit
        def empty_with_dtype(
            x: torch.Tensor,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return torch.empty_like(x, dtype=dtype).zero_()

        x = torch.randn(4, device="cuda")
        for dtype in (torch.float32, torch.float16):
            result = empty_with_dtype(x, dtype)
            self.assertEqual(result.dtype, dtype)
            torch.testing.assert_close(result, torch.zeros_like(result))

    def test_writeback_runs_once_on_initial_compile(self) -> None:
        @trident.jit
        def increment_in_place(x: torch.Tensor) -> torch.Tensor:
            x.add_(1)
            return x

        x = torch.randn(4, device="cuda")
        expected = x + 1
        result = increment_in_place(x)

        torch.testing.assert_close(x, expected)
        torch.testing.assert_close(result, expected)

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

    def test_nested_input_unpack(self) -> None:
        @trident.jit
        def nested_add(
            x: torch.Tensor,
            values: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> torch.Tensor:
            return x + values[0][0] + values[1][1]

        x = torch.randn(4, device="cuda")
        values = [
            (torch.randn(4, device="cuda"), torch.randn(4, device="cuda")),
            (torch.randn(4, device="cuda"), torch.randn(4, device="cuda")),
        ]
        torch.testing.assert_close(
            nested_add(x, values),
            x + values[0][0] + values[1][1],
        )


if __name__ == "__main__":
    unittest.main()
