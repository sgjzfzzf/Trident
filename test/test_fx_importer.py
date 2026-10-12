# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Direct FX import: aliases, symbolic scalars, containers and control flow."""

from __future__ import annotations

import operator
import unittest
from itertools import pairwise

import torch
import trident
from trident.core import ir, register_all_dialects
from trident.core.dialects import func
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.importer import TridentFxImporter
from trident.ir import box, value_type

from test.base import TridentTestCase


class FxImporterTest(TridentTestCase):
    def _import_graph(self, graph: torch.fx.Graph) -> TridentFxImporter:
        graph_module = torch.fx.GraphModule({}, graph)
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        output = next(node for node in graph.nodes if node.op == "output")
        result = output.args[0]
        context = ir.Context()
        register_all_dialects(context)
        importer = TridentFxImporter(context=context, specialization_id=0)
        with context, ir.Location.unknown(context):
            with ir.InsertionPoint(importer.module.body):
                function = func.FuncOp(
                    "graph",
                    ir.FunctionType.get(
                        [value_type(node.meta["val"]) for node in placeholders],
                        [value_type(result.meta["val"])],
                    ),
                )
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                for node, value in zip(placeholders, block.arguments, strict=True):
                    importer._bind(node, value)
                importer._import_nodes(graph, graph_module)
                func.ReturnOp([box(importer.argument(result))])
            importer.module.operation.verify()
        return importer

    def _registry_call(
        self, importer: TridentFxImporter, name: str
    ) -> tvm_ffi_d.FunctionCallOp:
        function = importer.module.body.operations[0]
        return next(
            operation
            for operation in function.body.blocks[0].operations
            if isinstance(operation, tvm_ffi_d.FunctionCallOp)
            and ir.StringAttr(operation.callee.owner.attributes["name"]).value == name
        )

    def test_aten_factory_arguments(self) -> None:
        graph = torch.fx.Graph()
        result = graph.call_function(
            torch.ops.aten.empty.memory_format,
            ([2, 3],),
            {
                "dtype": torch.float64,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "memory_format": torch.contiguous_format,
            },
        )
        result.meta["val"] = torch.empty(2, 3, dtype=torch.float64)
        graph.output(result)
        importer = self._import_graph(graph)
        call = self._registry_call(importer, "trident.aten.empty.memory_format")
        dtype = self._registry_call(importer, "trident.runtime.tvm_ffi_to_torch_type")
        shape = self._registry_call(importer, "ffi.Array")
        self.assertEqual(len(call.arguments), 6)
        self.assertEqual(call.arguments[0], shape.result)
        self.assertEqual(call.arguments[1], dtype.result)
        self.assertEqual(dtype.arguments[0].owner.name, "tvm_ffi.constant.dtype")
        for index in (2, 5):
            value = call.arguments[index].owner.operands[0]
            self.assertEqual(ir.IntegerAttr(value.owner.attributes["value"]).value, 0)
        self.assertEqual(call.arguments[3].owner.name, "tvm_ffi.constant.device")
        self.assertEqual(call.arguments[4].owner.name, "tvm_ffi.constant.none")

    def test_aten_scalar_arguments(self) -> None:
        for kwargs, alpha in (({}, 1), ({"alpha": 3}, 3)):
            with self.subTest(kwargs=kwargs):
                graph = torch.fx.Graph()
                tensor = graph.placeholder("tensor")
                tensor.meta["val"] = torch.empty(2)
                scalar = graph.placeholder("scalar")
                scalar.meta["val"] = 0.5
                result = graph.call_function(
                    torch.ops.aten.add.Tensor, (tensor, scalar), kwargs
                )
                result.meta["val"] = torch.empty(2)
                graph.output(result)
                importer = self._import_graph(graph)
                call = self._registry_call(importer, "trident.aten.add.Scalar")
                self.assertEqual(call.arguments[0], importer.values[tensor])
                self.assertEqual(call.arguments[1], importer.values[scalar])
                value = call.arguments[2].owner.operands[0]
                self.assertEqual(
                    ir.IntegerAttr(value.owner.attributes["value"]).value, alpha
                )

    def test_binary_operations(self) -> None:
        cases = (
            (max, "arith.maxsi", 0),
            (min, "arith.minsi", 0),
            (operator.add, "arith.addi", 0),
            (operator.eq, "arith.cmpi", False),
            (operator.floordiv, "arith.floordivsi", 0),
            (torch.sym_max, "arith.maxsi", 0),
            (torch.sym_min, "arith.minsi", 0),
        )
        for target, operation_name, metadata in cases:
            with self.subTest(target=target):
                graph = torch.fx.Graph()
                lhs = graph.placeholder("lhs")
                rhs = graph.placeholder("rhs")
                lhs.meta["val"], rhs.meta["val"] = -7, 3
                result = graph.call_function(target, (lhs, rhs))
                result.meta["val"] = metadata
                graph.output(result)
                importer = self._import_graph(graph)
                imported = importer.argument(result)
                self.assertEqual(imported.owner.name, operation_name)
                self.assertEqual(
                    imported.owner.operands[0].owner.operands[0], importer.values[lhs]
                )
                self.assertEqual(
                    imported.owner.operands[1].owner.operands[0], importer.values[rhs]
                )

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

    def test_integer_power(self) -> None:
        for exponent in (0, 1, 3):
            with self.subTest(exponent=exponent):
                graph = torch.fx.Graph()
                base = graph.placeholder("base")
                base.meta["val"] = 2
                result = graph.call_function(operator.pow, (base, exponent))
                result.meta["val"] = 2**exponent
                graph.output(result)
                importer = self._import_graph(graph)
                operations = list(
                    importer.module.body.operations[0].body.blocks[0].operations
                )
                extractions = [
                    op for op in operations if isinstance(op, tvm_ffi_d.GetOp)
                ]
                products = [op for op in operations if op.name == "arith.muli"]
                self.assertEqual(len(extractions), bool(exponent))
                self.assertEqual(len(products), exponent)
                if exponent:
                    self.assertEqual(extractions[0].operand, importer.values[base])
                    for product in products:
                        self.assertEqual(product.operands[1], extractions[0].result)
                    for previous, product in pairwise(products):
                        self.assertEqual(product.operands[0], previous.result)
                else:
                    imported = importer.argument(result)
                    self.assertEqual(
                        ir.IntegerAttr(imported.owner.attributes["value"]).value, 1
                    )

    def test_invalid_integer_power(self) -> None:
        for exponent in (-1, 0.5):
            with self.subTest(exponent=exponent):
                graph = torch.fx.Graph()
                result = graph.call_function(operator.pow, (2, exponent))
                result.meta["val"] = 0
                graph.output(result)
                with self.assertRaisesRegex(
                    NotImplementedError, "nonnegative constant exponent"
                ):
                    self._import_graph(graph)

    def test_missing_aten_argument(self) -> None:
        graph = torch.fx.Graph()
        tensor = graph.placeholder("tensor")
        tensor.meta["val"] = torch.empty(2)
        result = graph.call_function(torch.ops.aten.add.Tensor, (tensor,))
        result.meta["val"] = torch.empty(2)
        graph.output(result)
        with self.assertRaisesRegex(ValueError, "missing argument other"):
            self._import_graph(graph)

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

    def test_symbolic_integer_power(self) -> None:
        @trident.jit(dynamic=True)
        def power(x: torch.Tensor) -> int:
            return x.shape[0] ** 3

        for size in (3, 5, 7):
            x = torch.randn(size, device="cuda")
            self.assertEqual(power(x), size**3)
            self.assertEqual(power(x), size**3)
        self.assertEqual(len(power._sub_modules), 1)

    def test_symbolic_expression_power(self) -> None:
        import sympy

        context = ir.Context()
        register_all_dialects(context)
        importer = TridentFxImporter(context=context, specialization_id=0)
        symbol = sympy.Symbol("size", integer=True)
        with context, ir.Location.unknown(context):
            i64 = ir.IntegerType.get_signless(64)
            with ir.InsertionPoint(importer.module.body):
                function = func.FuncOp("power", ir.FunctionType.get([i64], [i64]))
                block = function.add_entry_block()
            with ir.InsertionPoint(block):
                importer.symbols[symbol] = block.arguments[0]
                result = importer.symbolic(sympy.Pow(symbol, 3, evaluate=False))
                func.ReturnOp([result])
            products = [op for op in block.operations if op.name == "arith.muli"]
            self.assertEqual(len(products), 3)
            for product in products:
                self.assertEqual(product.operands[1], block.arguments[0])
            self.assertEqual(result, products[-1].result)
            importer.module.operation.verify()

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

    def test_tensor_metadata(self) -> None:
        for target, arguments, expected, operation in (
            (torch.ops.aten.dim.default, (), 2, "tvm_ffi.tensor.dim"),
            (torch.ops.aten.sym_size.int, (0,), 2, "tvm_ffi.tensor.size"),
            (
                torch.ops.aten.sym_storage_offset.default,
                (),
                0,
                "tvm_ffi.tensor.storage_offset",
            ),
            (torch.ops.aten.sym_stride.int, (0,), 3, "tvm_ffi.tensor.stride"),
        ):
            with self.subTest(target=target):
                graph = torch.fx.Graph()
                tensor = graph.placeholder("tensor")
                tensor.meta["val"] = torch.empty(2, 3)
                result = graph.call_function(target, (tensor, *arguments))
                result.meta["val"] = expected
                graph.output(result)
                importer = self._import_graph(graph)
                self.assertEqual(importer.values[result].owner.name, operation)
                self.assertEqual(importer.argument(tensor), importer.values[tensor])

    def test_unsupported_call(self) -> None:
        graph = torch.fx.Graph()
        result = graph.call_function(operator.matmul, (2, 3))
        result.meta["val"] = 0
        graph.output(result)
        with self.assertRaisesRegex(NotImplementedError, "unsupported FX call.*matmul"):
            self._import_graph(graph)


if __name__ == "__main__":
    unittest.main()
