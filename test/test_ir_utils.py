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
from trident.core.dialects import tvm_ffi as ffi
from trident.ir_utils import binary, box, value_type


class ScalarTest(unittest.TestCase):
    def test_runtime_arithmetic(self) -> None:
        cases = (
            [
                (operation, lhs, rhs)
                for operation in (
                    operator.add,
                    operator.sub,
                    operator.mul,
                    operator.truediv,
                    operator.eq,
                    operator.ne,
                    operator.lt,
                    operator.le,
                    operator.gt,
                    operator.ge,
                    min,
                    max,
                )
                for lhs, rhs in ((-7, 3), (1.5, 2.0), (True, 2))
                if operation not in (min, max) or type(lhs) is type(rhs)
            ]
            + [
                (operation, -7, 3)
                for operation in (
                    operator.floordiv,
                    operator.mod,
                    operator.and_,
                    operator.or_,
                    operator.xor,
                    operator.lshift,
                    operator.rshift,
                )
            ]
            + [(operator.lt, False, True), (operator.gt, True, False)]
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
                    function = ffi.FuncOp(
                        f"scalar_{index}",
                        ir.TypeAttr.get(ir.FunctionType.get(inputs, [output])),
                        emit_tvm_ffi_abi=ir.UnitAttr.get(),
                    )
                    block = ir.Block.create_at_start(function.body, inputs)
                with ir.InsertionPoint(block):
                    result = binary(operation, *block.arguments)
                    ffi.ReturnOp([box(result)])
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


if __name__ == "__main__":
    unittest.main()
