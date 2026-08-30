# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch.export.graph_signature import (
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)
from trident.core import ir, register_all_dialects
from trident.core.dialects import func
from trident.input import InputNodeBuilder, InputTable, InputTableBuilder


class InputTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ir.Context()
        register_all_dialects(self.context)
        with self.context:
            self.tensor_type = ir.Type.parse("!torch.tensor")

        def target(
            x: torch.Tensor,
            scale: int,
            nested: list[tuple[torch.Tensor]],
            empty: list[torch.Tensor],
        ) -> torch.Tensor:
            return x * scale + nested[0][0]

        self.signature = inspect.signature(target)
        x = torch.randn(2)
        scale = 2
        nested = [(torch.randn(2),), (torch.randn(2),)]
        empty = []
        example_inputs = ((x, scale, nested, empty), {})
        _, in_spec = pytree.tree_flatten(example_inputs)
        graph_signature = ExportGraphSignature(
            input_specs=[
                InputSpec(InputKind.USER_INPUT, TensorArgument("x"), None),
                InputSpec(
                    InputKind.USER_INPUT,
                    ConstantArgument("scale", scale),
                    None,
                ),
                InputSpec(InputKind.USER_INPUT, TensorArgument("nested_0_0"), None),
                InputSpec(InputKind.USER_INPUT, TensorArgument("nested_1_0"), None),
            ],
            output_specs=[],
        )
        exported_program = SimpleNamespace(
            call_spec=SimpleNamespace(in_spec=in_spec),
            example_inputs=example_inputs,
            graph_signature=graph_signature,
        )
        self.builder = InputTableBuilder.get(
            exported_program,  # type: ignore[arg-type]
            self.signature,
            self.signature.bind(x, scale, nested, empty).arguments,
            [self.tensor_type] * 3,
            self._value_type,
            self.context,
        )

    def _value_type(self, value: Any) -> ir.Type:
        self.assertEqual(value, 2)
        return ir.Type.parse("!tvm_ffi.int", context=self.context)

    def _create_table(self) -> tuple[ir.Module, InputTable, ir.Block]:
        with self.context, ir.Location.unknown(self.context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    "resolve_input",
                    ir.FunctionType.get(self.builder.input_types, []),
                )
                block = function.add_entry_block()
            table = self.builder.build(block.arguments)
        return module, table, block

    def test_builder_exposes_wrapper_types_and_validates_operands(self) -> None:
        self.assertEqual(
            [str(type) for type in self.builder.input_types],
            [
                "!torch.tensor",
                "!tvm_ffi.int",
                "!tvm_ffi.array",
                "!tvm_ffi.array",
            ],
        )
        with self.assertRaisesRegex(AssertionError, "operand count"):
            self.builder.build([])

    def test_table_resolves_a_path_recursively(self) -> None:
        module, table, block = self._create_table()
        with (
            self.context,
            ir.Location.unknown(self.context),
            ir.InsertionPoint(block),
        ):
            value = table[("nested", 1, 0)]
            func.ReturnOp([])

        self.assertIsNotNone(value)
        self.assertEqual(value.type, self.tensor_type)
        self.assertEqual(str(module).count("tvm_ffi.array.get_item"), 2)

    def test_table_caches_materialized_children(self) -> None:
        module, table, block = self._create_table()
        with (
            self.context,
            ir.Location.unknown(self.context),
            ir.InsertionPoint(block),
        ):
            first = table[("nested", 1, 0)]
            second = table[("nested", 1, 0)]
            func.ReturnOp([])

        self.assertIs(first, second)
        self.assertEqual(str(module).count("tvm_ffi.array.get_item"), 2)

    def test_builder_creates_table_owned_nodes(self) -> None:
        _, first, block = self._create_table()
        second = self.builder.build(block.arguments)

        self.assertIsNot(first._nodes["nested"], second._nodes["nested"])
        self.assertIsNot(
            first._nodes["nested"].child(0),
            second._nodes["nested"].child(0),
        )

    def test_builder_does_not_materialize_child_values(self) -> None:
        module, _, block = self._create_table()
        with (
            self.context,
            ir.Location.unknown(self.context),
            ir.InsertionPoint(block),
        ):
            func.ReturnOp([])

        self.assertNotIn("tvm_ffi.array.get_item", str(module))

    def test_non_leaf_skips_empty_child_slots(self) -> None:
        with self.context, ir.Location.unknown(self.context):
            array_type = ir.Type.parse("!tvm_ffi.array")
            builder = InputNodeBuilder(
                array_type,
                children=[None, InputNodeBuilder(self.tensor_type)],
            )
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    "resolve_sparse_input",
                    ir.FunctionType.get([array_type], []),
                )
                block = function.add_entry_block()
            node = builder.build(block.arguments[0])
            with ir.InsertionPoint(block):
                value = node.resolve([1])
                self.assertIsNone(node.resolve([0]))
                flattened = node.flatten()
                func.ReturnOp([])

        self.assertEqual(flattened, [value])
        self.assertEqual(str(module).count("tvm_ffi.array.get_item"), 1)

    def test_table_handles_missing_and_invalid_paths(self) -> None:
        _, table, block = self._create_table()
        self.assertIsNone(table[("missing",)])
        with (
            self.context,
            ir.Location.unknown(self.context),
            ir.InsertionPoint(block),
        ):
            self.assertIsNone(table[("nested", 2)])
            self.assertIsNone(table[("x", 0)])

    def test_table_flattens_exported_inputs_in_graph_order(self) -> None:
        _, table, block = self._create_table()
        with (
            self.context,
            ir.Location.unknown(self.context),
            ir.InsertionPoint(block),
        ):
            inputs = table.flatten_inputs()
            func.ReturnOp([])

        self.assertEqual(
            [str(value.type) for value in inputs],
            ["!torch.tensor", "!tvm_ffi.int", "!torch.tensor", "!torch.tensor"],
        )


if __name__ == "__main__":
    unittest.main()
