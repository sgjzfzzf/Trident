# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Numerical validation for native frontend scalar arithmetic."""

from __future__ import annotations

import operator
import unittest

import tvm_ffi as runtime
from trident import capi_utils
from trident.core import (
    execution_engine,
    ir,
    passmanager,
    register_all_dialects,
    register_all_passes,
)
from trident.core.dialects import arith
from trident.core.dialects import tvm_ffi as tvm_ffi_d
from trident.ir import (
    _wrap_bool,
    _wrap_floating,
    _wrap_integer,
    binary,
    box,
    native,
    native_constant,
    value_type,
)


class ScalarTest(unittest.TestCase):
    def test_boolean_conversion(self) -> None:
        context = ir.Context()
        register_all_dialects(context)
        register_all_passes()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            inputs = [tvm_ffi_d.IntType.get(), tvm_ffi_d.FloatType.get()]
            with ir.InsertionPoint(module.body):
                function = tvm_ffi_d.FuncOp(
                    "boolean",
                    ir.TypeAttr.get(
                        ir.FunctionType.get(inputs, [tvm_ffi_d.BoolType.get()])
                    ),
                    emit_tvm_ffi_abi=ir.UnitAttr.get(),
                )
                block = ir.Block.create_at_start(function.body, inputs)
            with ir.InsertionPoint(block):
                result = _wrap_bool(arith.andi)(
                    *(native(value) for value in block.arguments)
                )
                tvm_ffi_d.ReturnOp([box(result)])
            module.operation.verify()
            passmanager.PassManager.parse(
                "builtin.module(trident-lowering-pipeline)"
            ).run(module.operation)
        engine = execution_engine.ExecutionEngine(
            module, shared_libs=capi_utils.find_runtime_libraries()
        )
        engine.initialize()
        function = runtime.Function.__from_mlir_packed_safe_call__(
            engine.raw_lookup("__tvm_ffi_boolean"), keep_alive_object=engine
        )
        for lhs, rhs in ((2, 2.0), (0, 2.0), (2, 0.0), (2, float("nan"))):
            with self.subTest(lhs=lhs, rhs=rhs):
                self.assertEqual(function(lhs, rhs), bool(lhs) and bool(rhs))

    def test_invalid_floating_operations(self) -> None:
        context = ir.Context()
        register_all_dialects(context)
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                for operation in (
                    operator.and_,
                    operator.floordiv,
                    operator.lshift,
                    operator.mod,
                    operator.or_,
                    operator.rshift,
                    operator.xor,
                ):
                    for lhs, rhs in ((1.5, 2), (2, 1.5), (1.5, 2.0)):
                        with (
                            self.subTest(operation=operation, lhs=lhs, rhs=rhs),
                            self.assertRaises(NotImplementedError),
                        ):
                            binary(
                                operation,
                                native_constant(lhs),
                                native_constant(rhs),
                            )

    def test_runtime_arithmetic(self) -> None:
        cases = (
            [
                (operation, lhs, rhs)
                for operation in (
                    max,
                    min,
                    operator.add,
                    operator.eq,
                    operator.ge,
                    operator.gt,
                    operator.le,
                    operator.lt,
                    operator.mul,
                    operator.ne,
                    operator.sub,
                    operator.truediv,
                )
                for lhs, rhs in (
                    (-7, 3),
                    (1.5, 2.0),
                    (True, 2),
                    (-7, 2.5),
                    (2.5, -7),
                    (True, 2.5),
                    (2.5, True),
                )
                if operation not in (min, max) or type(lhs) is type(rhs)
            ]
            + [
                (operation, -7, 3)
                for operation in (
                    operator.and_,
                    operator.floordiv,
                    operator.lshift,
                    operator.mod,
                    operator.or_,
                    operator.rshift,
                    operator.xor,
                )
            ]
            + [
                (operation, lhs, rhs)
                for operation in (
                    operator.and_,
                    operator.eq,
                    operator.ne,
                    operator.or_,
                    operator.xor,
                )
                for lhs, rhs in (
                    (False, False),
                    (False, True),
                    (True, False),
                    (True, True),
                )
            ]
            + [
                (operator.lt, False, True),
                (operator.gt, True, False),
                (operator.mod, 7, -3),
                (operator.mod, -7, -3),
            ]
        )
        context = ir.Context()
        register_all_dialects(context)
        register_all_passes()
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            for index, (operation, lhs, rhs) in enumerate(cases):
                inputs = [value_type(lhs), value_type(rhs)]
                output = value_type(operation(lhs, rhs))
                with ir.InsertionPoint(module.body):
                    function = tvm_ffi_d.FuncOp(
                        f"scalar_{index}",
                        ir.TypeAttr.get(ir.FunctionType.get(inputs, [output])),
                        emit_tvm_ffi_abi=ir.UnitAttr.get(),
                    )
                    block = ir.Block.create_at_start(function.body, inputs)
                with ir.InsertionPoint(block):
                    result = binary(operation, *block.arguments)
                    tvm_ffi_d.ReturnOp([box(result)])
            module.operation.verify()
            passmanager.PassManager.parse(
                "builtin.module(trident-lowering-pipeline)"
            ).run(module.operation)
        engine = execution_engine.ExecutionEngine(
            module, shared_libs=capi_utils.find_runtime_libraries()
        )
        engine.initialize()
        for index, (operation, lhs, rhs) in enumerate(cases):
            with self.subTest(operation=operation, lhs=lhs, rhs=rhs):
                function = runtime.Function.__from_mlir_packed_safe_call__(
                    engine.raw_lookup(f"__tvm_ffi_scalar_{index}"),
                    keep_alive_object=engine,
                )
                self.assertEqual(function(lhs, rhs), operation(lhs, rhs))

    def test_wrapper_operand_types(self) -> None:
        context = ir.Context()
        register_all_dialects(context)
        with context, ir.Location.unknown(context):
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                for wrapper, expected in (
                    (_wrap_bool, ir.IntegerType.get_signless(1)),
                    (_wrap_floating, ir.F64Type.get()),
                    (_wrap_integer, ir.IntegerType.get_signless(64)),
                ):

                    def builder(
                        lhs: ir.Value, rhs: ir.Value, expected: ir.Type = expected
                    ) -> ir.Value:
                        self.assertEqual(lhs.type, expected)
                        self.assertEqual(rhs.type, expected)
                        return lhs

                    wrapper(builder)(native_constant(True), native_constant(2))
                with self.assertRaises(NotImplementedError):
                    _wrap_integer(builder)(native_constant(1.5), native_constant(2))


if __name__ == "__main__":
    unittest.main()
