# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Frontend compilation tests."""

from __future__ import annotations

import unittest

import torch
import trident

from test.base import TridentTestCase


class FrontendTest(TridentTestCase):
    def test_container_inputs_unpack(self) -> None:
        @trident.jit
        def list_add(x: torch.Tensor, values: list[torch.Tensor]) -> torch.Tensor:
            return x + values[0] + values[1]

        @trident.jit
        def nested_add(
            x: torch.Tensor,
            values: list[tuple[torch.Tensor, torch.Tensor]],
        ) -> torch.Tensor:
            return x + values[0][0] + values[1][1]

        @trident.jit
        def tuple_add(
            x: torch.Tensor,
            values: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            return x + values[0] + values[1]

        x = torch.randn(4, device="cuda")
        first = torch.randn(4, device="cuda")
        second = torch.randn(4, device="cuda")
        for name, function, values, expected in (
            ("list", list_add, [first, second], x + first + second),
            (
                "nested",
                nested_add,
                [(first, torch.zeros_like(x)), (torch.zeros_like(x), second)],
                x + first + second,
            ),
            ("tuple", tuple_add, (first, second), x + first + second),
        ):
            with self.subTest(container=name):
                torch.testing.assert_close(function(x, values), expected)

    def test_argument_specialization(self) -> None:
        @trident.jit
        def empty_on_device(
            x: torch.Tensor,
            device: torch.device,
        ) -> torch.Tensor:
            return torch.empty_like(x, device=device).zero_()

        @trident.jit
        def empty_with_dtype(
            x: torch.Tensor,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            return torch.empty_like(x, dtype=dtype).zero_()

        x: torch.Tensor = torch.randn(4, device="cuda")
        device: torch.device = torch.device("cuda")
        first: torch.Tensor = empty_on_device(x, device)
        repeated: torch.Tensor = empty_on_device(x, device)
        torch.testing.assert_close(first, torch.zeros_like(x))
        torch.testing.assert_close(repeated, torch.zeros_like(x))

        cpu_result: torch.Tensor = empty_on_device(x, torch.device("cpu"))
        self.assertEqual(cpu_result.device, torch.device("cpu"))
        torch.testing.assert_close(cpu_result, torch.zeros_like(cpu_result))

        for dtype in (torch.float32, torch.float16):
            result: torch.Tensor = empty_with_dtype(x, dtype)
            self.assertEqual(result.dtype, dtype)
            torch.testing.assert_close(result, torch.zeros_like(result))
        self.assertEqual(len(empty_with_dtype._sub_modules), 2)

    def test_cached_result_and_writeback(self) -> None:
        @trident.jit
        def create_arange() -> torch.Tensor:
            return torch.arange(0, 64, 2, dtype=torch.float32, device="cuda")

        @trident.jit
        def increment_in_place(x: torch.Tensor) -> torch.Tensor:
            x.add_(1)
            return x

        expected = torch.arange(0, 64, 2, dtype=torch.float32, device="cuda")
        first = create_arange()
        second = create_arange()

        self.assertEqual(len(create_arange._sub_modules), 1)
        wrapper_module = f"{create_arange._sub_modules[0]}"
        self.assertIn("torchext.cast", wrapper_module)
        self.assertIn("-> !tvm_ffi.any", wrapper_module)
        self.assertNotIn("!torch.union<", wrapper_module)
        self.assertIsInstance(first, torch.Tensor)
        self.assertIsInstance(second, torch.Tensor)
        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second.cpu(), expected.cpu())

        x = torch.randn(4, device="cuda")
        expected_writeback = x + 1
        result = increment_in_place(x)

        torch.testing.assert_close(x, expected_writeback)
        torch.testing.assert_close(result, expected_writeback)


if __name__ == "__main__":
    unittest.main()
