# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Focused tests for translating Dynamo guards to Trident guard IR."""

from __future__ import annotations

import ast
import unittest
from dataclasses import dataclass

from torch._guards import GuardSource
from trident.core import ir, register_all_dialects
from trident.guards.codes import (
    ASTCode,
    ConstantCode,
    DynamoAttributeAbsentCode,
    GuardCode,
    RequiresGradCode,
    TensorDeviceCode,
    TensorDTypeCode,
    TensorRankCode,
    TypeIdCode,
)
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

from test.base import TridentTestCase


@dataclass(frozen=True)
class FakeGuard:
    create_fn: str
    code_list: list[str] | None
    name: str = ""
    source: GuardSource = GuardSource.LOCAL

    def create_fn_name(self) -> str:
        return self.create_fn


class GuardTest(TridentTestCase):
    def assert_parsed(
        self,
        guard: FakeGuard,
        kind: type[Guard],
        code_types: tuple[type[GuardCode], ...],
    ) -> Guard:
        parsed = Guard.parse(guard)
        self.assertIsInstance(parsed, kind)
        assert parsed is not None
        self.assertEqual(tuple(type(item.code) for item in parsed.codes), code_types)
        return parsed

    def build_ast_code(self, text: str) -> tuple[str, str]:
        """Insert a guard expression into a textually parsed IR fixture."""
        context = ir.Context()
        register_all_dialects(context)
        with context, ir.Location.unknown(context):
            module = ir.Module.parse(
                """
                module {
                  func.func @guard_expression() {
                    return
                  }
                }
                """
            )
            function = module.body.operations[0]
            block = function.regions[0].blocks[0]
            with ir.InsertionPoint(block.operations[-1]):
                code = ASTCode(text, ast.parse(text, mode="eval").body)
                result = code.build(None, context)  # type: ignore[arg-type]
            result_type = str(result.type)
            module_text = str(module)
        return result_type, module_text

    def test_ast_lowering(self) -> None:
        cases = (
            (
                "(1 | 2) == 3",
                "!torch.bool",
                ("arith.ori",),
                ("torch.aten.__or__.Scalar",),
            ),
            ("1 == 1 <= 3", "!torch.bool", ("torch.aten.le.int",), ()),
            (
                "1.0 + 2.0 <= 4.0",
                "!torch.bool",
                ("torch.aten.add.float", "torch.aten.ge.float"),
                (),
            ),
            ("1 + 2", "!torch.int", ("torch.aten.add.int",), ()),
            ("5 / 2", "!torch.float", ("torch.aten.div.int",), ()),
            ("True and not False", "!torch.bool", ("torch.aten.__not__",), ()),
            ("-1", "!torch.int", ("torch.aten.neg.int",), ()),
            ("torch.float32", "!torchext.dtype", ("#torchext.float32",), ()),
        )
        for text, result_type, expected, excluded in cases:
            with self.subTest(text=text):
                actual_type, module = self.build_ast_code(text)
                self.assertEqual(actual_type, result_type)
                for operation in expected:
                    self.assertIn(operation, module)
                for operation in excluded:
                    self.assertNotIn(operation, module)

    def test_constant_match(self) -> None:
        supported = (
            "L['value'] is None",
            "L['value'] == True",
            "L['value'] == -7",
            "L['value'] == 1.5",
            "L['value'] == 'constant'",
            "L['value'] == device(type='cuda', index=1)",
            "L['value'] == torch.float32",
        )
        for text in supported:
            with self.subTest(text=text):
                parsed = self.assert_parsed(
                    FakeGuard("CONSTANT_MATCH", [text], "L['value']"),
                    ConstantMatchGuard,
                    (ConstantCode,),
                )
                code = parsed.codes[0].code
                assert isinstance(code, ConstantCode)
                self.assertEqual(
                    ast.dump(code.expression, include_attributes=False),
                    ast.dump(
                        ast.parse(text, mode="eval").body, include_attributes=False
                    ),
                )

        rejected = (
            "L['value'] == [1, 2]",
            "L['value'] == device('cpu')",
            "L['value'] != 1",
            "1 == L['value']",
            "G['value'] == 1",
        )
        for text in rejected:
            with self.subTest(text=text):
                self.assertIsNone(
                    Guard.parse(FakeGuard("CONSTANT_MATCH", [text], "L['value']"))
                )

    def test_guard_kinds(self) -> None:
        cases = (
            (
                FakeGuard("DUPLICATE_INPUT", ["L['x'] is L['y']"], "L['x']"),
                DuplicateInputGuard,
                (ASTCode,),
            ),
            (FakeGuard("GLOBAL_STATE", None), IgnoredGuard, ()),
            (
                FakeGuard(
                    "SEQUENCE_LENGTH",
                    [
                        "___check_type_id(L['xs'], 1), type=<class 'list'>",
                        "len(L['xs']) == 2",
                    ],
                    "L['xs']",
                ),
                SequenceLengthGuard,
                (TypeIdCode, ASTCode),
            ),
            (
                FakeGuard(
                    "SHAPE_ENV",
                    [
                        "2 <= L['x'].size()[0] <= 8",
                        "L['x'].size()[1] == 4 * L['y'].size()[1]",
                    ],
                ),
                ShapeEnvGuard,
                (ASTCode, ASTCode),
            ),
            (
                FakeGuard(
                    "TYPE_MATCH",
                    ["___check_type_id(L['xs'][0], 2), type=<class 'torch.Tensor'>"],
                    "L['xs'][0]",
                ),
                TypeMatchGuard,
                (TypeIdCode,),
            ),
        )
        for guard, kind, code_types in cases:
            with self.subTest(kind=guard.create_fn):
                parsed = self.assert_parsed(guard, kind, code_types)
                if guard.name:
                    self.assertEqual(parsed.source, Local.parse(guard.name))

    def test_rejected_guards_and_invariants(self) -> None:
        rejected = (
            FakeGuard("UNKNOWN", None),
            FakeGuard("SHAPE_ENV", ["L['x'].size()[0] == 2"], "L['x']"),
            FakeGuard("CONSTANT_MATCH", None, "L['x']"),
            FakeGuard("CONSTANT_MATCH", ["L['x'] == 1", "L['x'] == 2"], "L['x']"),
            FakeGuard("TENSOR_MATCH", ["L['x'].ndimension() >= 2"], "L['x']"),
            FakeGuard("TYPE_MATCH", ["___check_obj_id(L['x'], 1)"], "L['x']"),
        )
        for guard in rejected:
            with self.subTest(kind=guard.create_fn, codes=guard.code_list):
                self.assertIsNone(Guard.parse(guard))

        for text, message in (
            ("L['x'] is L['y'] is L['z']", "identity guard comparison must be binary"),
            ("1 in (1, 2)", "unsupported guard comparison operator"),
        ):
            with self.subTest(text=text):
                code = ASTCode(text, ast.parse(text, mode="eval").body)
                context = ir.Context()
                with (
                    context,
                    ir.Location.unknown(context),
                    self.assertRaisesRegex(AssertionError, message),
                ):
                    code.build(None, context)  # type: ignore[arg-type]

        with self.assertWarnsRegex(RuntimeWarning, "Tensor\\._base guard"):
            result_type, module = self.build_ast_code("L['x']._base.size()[0] == 1")
        self.assertEqual(result_type, "!torch.bool")
        self.assertIn("torch.constant.bool false", module)

        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            Guard.parse(
                FakeGuard(
                    "TENSOR_MATCH",
                    ["str(L['x'].device) == 'not-a-device'"],
                    "L['x']",
                )
            )

    def test_tensor_match(self) -> None:
        code_list = [
            "___check_type_id(L['xs'][0], 1), type=<class 'torch.Tensor'>",
            "str(L['xs'][0].dtype) == 'torch.float32'",
            "str(L['xs'][0].device) == 'cuda:1'",
            "L['xs'][0].requires_grad == True",
            "L['xs'][0].ndimension() == 4",
            "hasattr(L['xs'][0], '_dynamo_static_indices') == False",
        ]
        parsed = self.assert_parsed(
            FakeGuard("TENSOR_MATCH", code_list, "L['xs'][0]"),
            TensorMatchGuard,
            (
                TypeIdCode,
                TensorDTypeCode,
                TensorDeviceCode,
                RequiresGradCode,
                TensorRankCode,
                DynamoAttributeAbsentCode,
            ),
        )
        self.assertEqual(parsed.source, Local(("xs", 0)))
        _, dtype_item, device_item, grad_item, rank_item, attribute_item = parsed.codes
        dtype = dtype_item.code
        device = device_item.code
        grad = grad_item.code
        rank = rank_item.code
        attribute = attribute_item.code
        assert isinstance(dtype, TensorDTypeCode)
        assert isinstance(device, TensorDeviceCode)
        assert isinstance(grad, RequiresGradCode)
        assert isinstance(rank, TensorRankCode)
        assert isinstance(attribute, DynamoAttributeAbsentCode)
        self.assertEqual(dtype.expected, "torch.float32")
        self.assertEqual(device.expected, "cuda:1")
        self.assertTrue(grad.expected)
        self.assertEqual(rank.expected, 4)
        self.assertEqual(attribute.attribute, "_dynamo_static_indices")


if __name__ == "__main__":
    unittest.main()
