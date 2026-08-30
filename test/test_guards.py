# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Tests for mapping Dynamo Guards to Trident GuardCode objects."""

from __future__ import annotations

import ast
import unittest
from dataclasses import dataclass
from typing import Any

import torch
from base import TridentTestCase
from torch._guards import GuardSource
from trident.core import ir, register_all_dialects
from trident.core.dialects import func
from trident.guards.codes import (
    ConstantCode,
    DynamoAttributeAbsentCode,
    ExpressionCode,
    GuardCode,
    RequiresGradCode,
    SequenceLengthCode,
    TensorDeviceCode,
    TensorDTypeCode,
    TensorRankCode,
    TypeIdCode,
)
from trident.guards.collection import Guards
from trident.guards.kinds import (
    ConstantMatchGuard,
    DuplicateInputGuard,
    Guard,
    IgnoredGuard,
    SequenceLengthGuard,
    ShapeEnvGuard,
    TensorMatchGuard,
    TypeMatchGuard,
)
from trident.guards.local import Local
from trident.input import InputTable


@dataclass(frozen=True)
class FakeGuard:
    create_fn: str
    code_list: list[str] | None
    name: str = ""
    source: GuardSource = GuardSource.LOCAL

    def create_fn_name(self) -> str:
        return self.create_fn


class FirstMatchingCode(GuardCode):
    @classmethod
    def parse(cls, text: str, source: Local | None) -> FirstMatchingCode:
        return cls(text, source)

    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        return super().build(tree, context)


class SecondMatchingCode(FirstMatchingCode):
    pass


class AmbiguousTestGuard(Guard):
    create_fn_name = "TEST_AMBIGUOUS"
    code_types = (FirstMatchingCode, SecondMatchingCode)


class GuardParserTest(TridentTestCase):
    def test_build_creates_a_table_for_each_guard_region(self) -> None:
        guards = Guards(
            [
                FakeGuard(
                    "TYPE_MATCH",
                    ["___check_type_id(L['x'], 1), type=<class 'torch.Tensor'>"],
                    "L['x']",
                ),
                FakeGuard(
                    "TYPE_MATCH",
                    ["___check_type_id(L['y'], 2), type=<class 'torch.Tensor'>"],
                    "L['y']",
                ),
            ]
        )
        table_count = 0

        def table_factory() -> InputTable:
            nonlocal table_count
            table_count += 1
            return None  # type: ignore[return-value]

        context = ir.Context()
        register_all_dialects(context)
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    "guard_tables",
                    ir.FunctionType.get([], []),
                )
                block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    guards.build(table_factory, context)
                    func.ReturnOp([])

        self.assertEqual(table_count, 2)

    def assert_parsed(
        self,
        guard: FakeGuard,
        handler_type: type[Guard],
        code_types: tuple[type[GuardCode], ...],
    ) -> Guard:
        parsed = Guard.parse(guard)
        self.assertIsInstance(parsed, handler_type)
        assert parsed is not None
        self.assertEqual(tuple(map(type, parsed.codes)), code_types)
        return parsed

    def test_ignored_guards_accept_every_code_list_shape(self) -> None:
        cases = [
            (
                "AUTOGRAD_SAVED_TENSORS_HOOKS",
                ["top_saved_tensors_hooks ids == None"],
            ),
            ("DEFAULT_DEVICE", ["CURRENT_DEVICE == None"]),
            ("DETERMINISTIC_ALGORITHMS", None),
            ("GLOBAL_STATE", []),
            ("GRAD_MODE", ["first", "second"]),
            ("TORCH_FUNCTION_STATE", None),
        ]
        for create_fn_name, code_list in cases:
            with self.subTest(create_fn_name=create_fn_name, code_list=code_list):
                parsed = self.assert_parsed(
                    FakeGuard(create_fn_name, code_list),
                    IgnoredGuard,
                    (),
                )
                self.assertIsNone(parsed.source)

    def test_constant_match_parses_supported_literals(self) -> None:
        cases: list[tuple[str, Any]] = [
            ("L['value'] is None", None),
            ("L['value'] == True", True),
            ("L['value'] == False", False),
            ("L['value'] == -7", -7),
            ("L['value'] == 1.5", 1.5),
            ("L['value'] == 'constant'", "constant"),
            ("L['value'] == ''", ""),
            ("L['value'] == torch.float32", torch.float32),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                parsed = self.assert_parsed(
                    FakeGuard("CONSTANT_MATCH", [text], "L['value']"),
                    ConstantMatchGuard,
                    (ConstantCode,),
                )
                code = parsed.codes[0]
                assert isinstance(code, ConstantCode)
                self.assertEqual(code.value, expected)
                self.assertIs(type(code.value), type(expected))

    def test_duplicate_input_parses_identity_guard(self) -> None:
        parsed = self.assert_parsed(
            FakeGuard(
                "DUPLICATE_INPUT",
                ["L['out_grad'] is L['new_out']"],
                "L['new_out']",
            ),
            DuplicateInputGuard,
            (ExpressionCode,),
        )
        (code,) = parsed.codes
        assert isinstance(code, ExpressionCode)

    def test_duplicate_input_rejects_non_identity_expressions(self) -> None:
        for text in (
            "L['out_grad'] == L['new_out']",
            "L['out_grad'] is not L['new_out']",
            "L['out_grad'] is L['new_out'] or True",
        ):
            with self.subTest(text=text):
                self.assertIsNone(
                    Guard.parse(FakeGuard("DUPLICATE_INPUT", [text], "L['new_out']"))
                )

    def test_constant_match_rejects_unsupported_literals(self) -> None:
        for literal in ("b'constant'", "[1, 2]", "{'value': 1}", "not valid"):
            with self.subTest(literal=literal):
                self.assertIsNone(
                    Guard.parse(
                        FakeGuard(
                            "CONSTANT_MATCH", [f"L['value'] == {literal}"], "L['value']"
                        )
                    )
                )

    def test_constant_match_requires_exactly_one_code(self) -> None:
        invalid_code_lists = (
            None,
            [],
            ["L['value'] == 1", "L['value'] == 2"],
        )
        for code_list in invalid_code_lists:
            with self.subTest(code_list=code_list):
                self.assertIsNone(
                    Guard.parse(FakeGuard("CONSTANT_MATCH", code_list, "L['value']"))
                )

    def test_constant_match_rejects_invalid_operations_and_sources(self) -> None:
        cases = [
            ("L['value'] is 1", "L['value']"),
            ("L['value'] is 'constant'", "L['value']"),
            ("L['value'] != 1", "L['value']"),
            ("L['other'] == 1", "L['value']"),
            ("G['value'] == 1", "G['value']"),
        ]
        for text, name in cases:
            with self.subTest(text=text, name=name):
                self.assertIsNone(
                    Guard.parse(FakeGuard("CONSTANT_MATCH", [text], name))
                )

    def test_sequence_length_parses_type_length_and_empty_checks(self) -> None:
        cases = [
            (
                "L['xs']",
                [
                    "___check_type_id(L['xs'], 123), type=<class 'list'>",
                    "len(L['xs']) == 2",
                ],
                (TypeIdCode, SequenceLengthCode),
                2,
                0,
            ),
            (
                "L['xss'][1]",
                ["not L['xss'][1]"],
                (SequenceLengthCode,),
                0,
                1,
            ),
        ]
        for name, code_list, code_types, expected_length, expected_depth in cases:
            with self.subTest(name=name):
                parsed = self.assert_parsed(
                    FakeGuard("SEQUENCE_LENGTH", code_list, name),
                    SequenceLengthGuard,
                    code_types,
                )
                length_code = next(
                    code
                    for code in parsed.codes
                    if isinstance(code, SequenceLengthCode)
                )
                self.assertEqual(length_code.expected, expected_length)
                self.assertEqual(length_code.depth, expected_depth)

    def test_sequence_length_rejects_unsupported_code(self) -> None:
        cases = [
            "len(L['other']) == 2",
            "len(L['xs']) != 2",
            "len(L['xs']) == -1",
            "bool(L['xs']) == False",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(
                    Guard.parse(FakeGuard("SEQUENCE_LENGTH", [text], "L['xs']"))
                )

    def test_shape_env_accepts_none_empty_and_multiple_codes(self) -> None:
        cases = [
            (None, ()),
            ([], ()),
            (
                [
                    "L['x'].size()[0] == 2",
                    "L['x'].stride()[0] == 1",
                    "L['x'].storage_offset() == 0",
                    "L['x'].size()[0] == L['y'].size()[0]",
                    "1 if L['x'].ndimension() == 2 else 0",
                ],
                (ExpressionCode,) * 5,
            ),
        ]
        for code_list, code_types in cases:
            with self.subTest(code_list=code_list):
                self.assert_parsed(
                    FakeGuard("SHAPE_ENV", code_list),
                    ShapeEnvGuard,
                    code_types,
                )

    def test_shape_env_accepts_dynamic_shape_arithmetic_and_chained_comparisons(
        self,
    ) -> None:
        cases = [
            (
                ["2 <= L['x'].size()[0] <= 8"],
                (ExpressionCode,),
            ),
            (
                [
                    "L['x'].size()[0] == 2 * L['y'].size()[0]",
                    "L['x'].size()[1] == L['y'].size()[1] * 4",
                ],
                (ExpressionCode,) * 2,
            ),
            (
                [
                    "L['x'].size()[0] + 1 == 2 * L['y'].size()[0]",
                    "L['x'].size()[0] == 2 and L['x'].size()[1] >= 1",
                ],
                (ExpressionCode,) * 2,
            ),
            (
                [
                    "0 <= L['x'].size()[0]",
                    "L['x'].size()[0] <= 2 * L['y'].size()[0]",
                    "L['x'].size()[0] % 2 == 0",
                ],
                (ExpressionCode,) * 3,
            ),
        ]
        for code_list, code_types in cases:
            with self.subTest(code_list=code_list):
                self.assert_parsed(
                    FakeGuard("SHAPE_ENV", code_list),
                    ShapeEnvGuard,
                    code_types,
                )

    def test_shape_env_requires_an_empty_guard_name(self) -> None:
        self.assertIsNone(
            Guard.parse(
                FakeGuard(
                    "SHAPE_ENV",
                    ["L['x'].size()[0] == 2"],
                    "L['x']",
                )
            )
        )

    def test_tensor_match_parses_every_supported_code(self) -> None:
        code_list = [
            "___check_type_id(L['x'], 123), type=<class 'torch.Tensor'>",
            "str(L['x'].dtype) == 'torch.float32'",
            "str(L['x'].device) == 'cpu'",
            "L['x'].requires_grad == False",
            "L['x'].ndimension() == 2",
            "hasattr(L['x'], '_dynamo_dynamic_indices') == False",
            "hasattr(L['x'], '_dynamo_weak_dynamic_indices') == False",
            "hasattr(L['x'], '_dynamo_unbacked_indices') == False",
            "hasattr(L['x'], '_dynamo_strict_unbacked_indices') == False",
            "hasattr(L['x'], '_dynamo_static_indices') == False",
        ]
        parsed = self.assert_parsed(
            FakeGuard("TENSOR_MATCH", code_list, "L['x']"),
            TensorMatchGuard,
            (
                TypeIdCode,
                TensorDTypeCode,
                TensorDeviceCode,
                RequiresGradCode,
                TensorRankCode,
                *(DynamoAttributeAbsentCode,) * 5,
            ),
        )

        _, dtype, device, _, rank, *attributes = parsed.codes
        assert isinstance(dtype, TensorDTypeCode)
        assert isinstance(device, TensorDeviceCode)
        assert isinstance(rank, TensorRankCode)
        self.assertEqual(dtype.expected, "torch.float32")
        self.assertEqual(device.expected, "cpu")
        self.assertEqual(rank.expected, 2)
        self.assertEqual(
            tuple(
                code.attribute
                for code in attributes
                if isinstance(code, DynamoAttributeAbsentCode)
            ),
            (
                "_dynamo_dynamic_indices",
                "_dynamo_weak_dynamic_indices",
                "_dynamo_unbacked_indices",
                "_dynamo_strict_unbacked_indices",
                "_dynamo_static_indices",
            ),
        )

    def test_tensor_match_supports_nested_sources_and_alternate_metadata(self) -> None:
        cases = [
            (
                "str(L['xs'][0].dtype) == \"torch.int64\"",
                TensorDTypeCode,
            ),
            ("str(L['xs'][0].device) == 'cuda:1'", TensorDeviceCode),
            ("L['xs'][0].ndimension() == 4", TensorRankCode),
            ("L['xs'][0].requires_grad == False", RequiresGradCode),
            (
                "hasattr(L['xs'][0], '_dynamo_static_indices') == False",
                DynamoAttributeAbsentCode,
            ),
        ]
        for text, code_type in cases:
            with self.subTest(text=text):
                parsed = self.assert_parsed(
                    FakeGuard("TENSOR_MATCH", [text], "L['xs'][0]"),
                    TensorMatchGuard,
                    (code_type,),
                )
                self.assertEqual(parsed.source, Local(("xs", 0)))

    def test_tensor_match_parses_requires_grad_true(self) -> None:
        parsed = self.assert_parsed(
            FakeGuard(
                "TENSOR_MATCH",
                ["L['x'].requires_grad == True"],
                "L['x']",
            ),
            TensorMatchGuard,
            (RequiresGradCode,),
        )
        code = parsed.codes[0]
        assert isinstance(code, RequiresGradCode)
        self.assertTrue(code.expected)

    def test_requires_grad_builds_an_i1_true_value(self) -> None:
        code = RequiresGradCode(
            "L['x'].requires_grad == True",
            Local(("x",)),
            True,
        )
        context = ir.Context()
        with context, ir.Location.unknown(context):
            result = code.build(None, context)  # type: ignore[arg-type]
        self.assertEqual(str(result.type), "i1")

    def test_expression_with_base_is_skipped_with_warning(self) -> None:
        cases = [
            "L['x']._base.size()[0] == 1",
            "L['x']._base.stride()[0] == 1",
            "L['x']._base.storage_offset() == 0",
        ]
        for text in cases:
            with self.subTest(text=text):
                expression = ast.parse(text, mode="eval").body
                code = ExpressionCode(text, expression)
                context = ir.Context()
                with (
                    context,
                    ir.Location.unknown(context),
                    self.assertWarnsRegex(
                        RuntimeWarning,
                        "Skipping unsupported Tensor\\._base guard",
                    ),
                ):
                    result = code.build(None, context)  # type: ignore[arg-type]
                self.assertEqual(str(result.type), "i1")

    def test_chained_comparison_builds_an_i1_value(self) -> None:
        text = "1 == 1 <= 3"
        code = ExpressionCode(text, ast.parse(text, mode="eval").body)
        context = ir.Context()
        register_all_dialects(context)
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    "chained_comparison",
                    ir.FunctionType.get([], []),
                )
                block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    result = code.build(None, context)  # type: ignore[arg-type]
                    func.ReturnOp([])
        self.assertEqual(str(result.type), "i1")

    def test_tensor_identity_is_skipped_with_warning(self) -> None:
        text = "L['x'] is L['y']"
        code = ExpressionCode(text, ast.parse(text, mode="eval").body)
        context = ir.Context()
        register_all_dialects(context)
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                tensor_type = ir.Type.parse("!torch.tensor", context=context)
                function = func.FuncOp(
                    "tensor_identity",
                    ir.FunctionType.get([tensor_type, tensor_type], []),
                )
                block = function.add_entry_block()
                with ir.InsertionPoint(block):
                    func.ReturnOp([])
            [x, y] = block.arguments
            tree = {
                ("x",): x,
                ("y",): y,
            }
            with self.assertWarnsRegex(
                RuntimeWarning,
                "Skipping unsupported Tensor identity guard",
            ):
                result = code.build(tree, context)  # type: ignore[arg-type]
        self.assertEqual(str(result.type), "i1")

    def test_invalid_comparison_violates_guard_code_invariants(self) -> None:
        cases = [
            ("True == 1", "guard comparison operand cannot be lowered"),
            (
                "L['x'] is L['y'] is L['z']",
                "identity guard comparison must be binary",
            ),
            (
                "L['x'] == L['y'] is L['z']",
                "identity guard comparison must be binary",
            ),
            ("1 in (1, 2)", "unsupported guard comparison operator"),
        ]
        for text, message in cases:
            with self.subTest(text=text):
                code = ExpressionCode(text, ast.parse(text, mode="eval").body)
                context = ir.Context()
                with (
                    context,
                    ir.Location.unknown(context),
                    self.assertRaisesRegex(AssertionError, message),
                ):
                    code.build(None, context)  # type: ignore[arg-type]

    def test_tensor_match_rejects_unsupported_codes(self) -> None:
        cases = [
            "str(L['x'].dtype) == 'torch.not_a_dtype'",
            "str(L['x'].device) == 'not-a-device'",
            "L['x'].ndimension() >= 2",
            "hasattr(L['x'], 'custom_attribute') == False",
            "hasattr(L['other'], '_dynamo_dynamic_indices') == False",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertIsNone(
                    Guard.parse(FakeGuard("TENSOR_MATCH", [text], "L['x']"))
                )

    def test_type_match_parses_quote_and_source_variants(self) -> None:
        cases = [
            (
                "L['x']",
                "___check_type_id(L['x'], 123), type=<class 'torch.Tensor'>",
            ),
            (
                "L['xs'][0]",
                "___check_type_id(L['xs'][0], 456), type=<class \"list\">",
            ),
        ]
        for name, text in cases:
            with self.subTest(text=text):
                parsed = self.assert_parsed(
                    FakeGuard("TYPE_MATCH", [text], name),
                    TypeMatchGuard,
                    (TypeIdCode,),
                )
                self.assertEqual(parsed.source, Local.parse(name))

    def test_empty_code_lists_follow_non_constant_handler_contracts(self) -> None:
        cases = [
            ("SEQUENCE_LENGTH", SequenceLengthGuard),
            ("TENSOR_MATCH", TensorMatchGuard),
            ("TYPE_MATCH", TypeMatchGuard),
        ]
        for create_fn_name, handler_type in cases:
            with self.subTest(create_fn_name=create_fn_name):
                self.assert_parsed(
                    FakeGuard(create_fn_name, [], "L['x']"),
                    handler_type,
                    (),
                )

    def test_rejects_unknown_ambiguous_and_unmatched_guards(self) -> None:
        cases = [
            FakeGuard("UNKNOWN", None),
            FakeGuard("TEST_AMBIGUOUS", ["L['x'] == 1"], "L['x']"),
            FakeGuard("TYPE_MATCH", ["___check_obj_id(L['x'], 1)"], "L['x']"),
            FakeGuard(
                "TYPE_MATCH",
                ["___check_type_id(L['other'], 1), type=<class 'int'>"],
                "L['x']",
            ),
        ]
        for guard in cases:
            with self.subTest(create_fn_name=guard.create_fn_name()):
                self.assertIsNone(Guard.parse(guard))

    def test_duplicate_registration_fails(self) -> None:
        with self.assertRaisesRegex(AssertionError, "duplicate Guard registration"):

            class DuplicateTypeMatchGuard(Guard):
                create_fn_name = "TYPE_MATCH"


if __name__ == "__main__":
    unittest.main()
