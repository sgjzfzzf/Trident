# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Direct FX import: aliases, symbolic scalars, containers and control flow."""

from __future__ import annotations

import unittest

import torch
import trident
from trident.core import ir, register_all_dialects
from trident.fx_importer import TridentFxImporter

from test.base import TridentTestCase


class FxImporterTest(TridentTestCase):
    def test_cond(self) -> None:
        @trident.jit(dynamic=True)
        def conditional(x: torch.Tensor):
            return torch.cond(x.shape[0] > 4, lambda t: t + 1, lambda t: t - 1, (x,))

        for size in (3, 6, 8):
            x = torch.randn(size, device="cuda")
            expected = x + 1 if size > 4 else x - 1
            conditional(x)
            torch.testing.assert_close(conditional(x), expected)
        self.assertIn("scf.if", str(conditional._sub_modules[0]))

    def test_dynamic_shapes(self) -> None:
        @trident.jit(dynamic=True)
        def reshape(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0] // 2, x.shape[1] * 2)

        for size in (6, 10, 14):
            x = torch.randn(size, 3, device="cuda")
            torch.testing.assert_close(reshape(x), x.reshape(size // 2, 6))
            torch.testing.assert_close(reshape(x), x.reshape(size // 2, 6))
        text = str(reshape._sub_modules[0])
        self.assertIn("tvm_ffi.tensor.size", text)
        self.assertIn("arith.floordivsi", text)
        self.assertIn("arith.muli", text)
        self.assertNotIn("!torch.", text)
        self.assertEqual(len(reshape._sub_modules), 1)

    def test_inplace_alias(self) -> None:
        @trident.jit
        def increment(x: torch.Tensor) -> torch.Tensor:
            view = x.view(-1)
            x.add_(1)
            return view

        x = torch.randn(2, 3, device="cuda")
        expected = x + 2
        increment(x)
        result = increment(x)
        torch.testing.assert_close(x, expected)
        torch.testing.assert_close(result, expected.flatten())
        self.assertEqual(result.data_ptr(), x.data_ptr())
        text = str(increment._sub_modules[0])
        self.assertIn("trident.aten.add_.Scalar", text)
        self.assertNotIn("trident.aten.clone", text)
        self.assertNotIn("tensor.copy_", text)

    def test_multiple_aten_results(self) -> None:
        @trident.jit
        def topk(x: torch.Tensor):
            return torch.topk(x, 2)

        x = torch.randn(4, device="cuda")
        for _ in range(2):
            values, indices = topk(x)
            expected = torch.topk(x, 2)
            torch.testing.assert_close(values, expected.values)
            torch.testing.assert_close(indices, expected.indices)

    def test_nested_outputs(self) -> None:
        @trident.jit
        def outputs(x: torch.Tensor):
            return {"tensor": x + 1, "other": (x * 2, [3, None])}

        x = torch.randn(4, device="cuda")
        for _ in range(2):
            result = outputs(x)
            self.assertIsInstance(result, dict)
            self.assertIsInstance(result["other"], tuple)
            self.assertEqual(result["other"][1], [3, None])
            torch.testing.assert_close(result["tensor"], x + 1)
            torch.testing.assert_close(result["other"][0], x * 2)

    def test_symbolic_hint_is_rejected(self) -> None:
        import sympy

        context = ir.Context()
        register_all_dialects(context)
        importer = TridentFxImporter(context=context, specialization_id=0)
        with (
            context,
            ir.Location.unknown(context),
            self.assertRaisesRegex(NotImplementedError, "concrete hint"),
        ):
            importer.symbolic(sympy.Symbol("unbound", integer=True))


if __name__ == "__main__":
    unittest.main()
