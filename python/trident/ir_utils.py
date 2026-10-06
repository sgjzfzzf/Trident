# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Shared builders for the FX frontend and Dynamo guards.

Scalars stay native while computing; FFI calls and containers receive boxed
semantic values. Calls are effectful and check lookup, handle, and call status.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

import torch
import tvm_ffi as runtime

from trident.core import ir
from trident.core.dialects import arith, builtin, cf, llvm
from trident.core.dialects import tvm_ffi as ffi


def ffi_type(name: str) -> ir.Type:
    return ir.Type.parse(f"!tvm_ffi.{name}")


def value_type(value: Any) -> ir.Type:
    if isinstance(value, torch.Tensor):
        name = "tensor"
    elif isinstance(value, (bool, torch.SymBool)):
        name = "bool"
    elif isinstance(value, (int, torch.SymInt)):
        name = "int"
    elif isinstance(value, (float, torch.SymFloat)):
        name = "float"
    elif isinstance(value, str):
        return ffi_type("union<!tvm_ffi.raw_str, !tvm_ffi.small_str, !tvm_ffi.str>")
    elif isinstance(value, torch.device):
        name = "device"
    elif isinstance(value, torch.dtype):
        name = "dtype"
    elif isinstance(value, (tuple, list, dict)):
        name = "array"
    elif value is None:
        name = "none"
    else:
        raise NotImplementedError(f"unsupported FX value type: {type(value)}")
    return ffi_type(name)


def native_constant(value: bool | float) -> ir.Value:
    if isinstance(value, bool):
        type_ = ir.IntegerType.get_signless(1)
    elif isinstance(value, int):
        type_ = ir.IntegerType.get_signless(64)
        if not -(1 << 63) <= value < (1 << 63):
            raise OverflowError(f"integer does not fit the i64 runtime ABI: {value}")
    else:
        type_ = ir.F64Type.get()
    return arith.constant(type_, value)


def native(value: ir.Value) -> ir.Value:
    if str(value.type) in ("!tvm_ffi.bool", "!tvm_ffi.int", "!tvm_ffi.float"):
        return ffi.get(value)
    return value


def box(value: ir.Value) -> ir.Value:
    if str(value.type) in ("i1", "i64", "f64"):
        return ffi.to(value)
    return value


def checked_call(name: str, arguments: Sequence[ir.Value], result: ir.Type) -> ir.Value:
    handle = ffi.FunctionGetGlobalOp(name)
    cf.AssertOp(handle.success, f"TVMFFIFunctionGetGlobal failed for {name}")
    pointer = builtin.unrealized_conversion_cast(
        [ir.Type.parse("!llvm.ptr")], [handle.result]
    )
    null = llvm.mlir_zero(ir.Type.parse("!llvm.ptr"))
    nonnull = llvm.icmp(llvm.ICmpPredicate.ne, pointer, null)
    cf.AssertOp(nonnull, f"TVMFFIFunctionGetGlobal returned null for {name}")
    call = ffi.FunctionCallOp(
        result,
        ir.IntegerType.get_signless(1),
        handle.result,
        [box(argument) for argument in arguments],
    )
    cf.AssertOp(call.success, f"TVMFFIFunctionCall failed for {name}")
    return call.result


def array(values: Sequence[ir.Value]) -> ir.Value:
    return checked_call("ffi.Array", values, ffi_type("array"))


def array_get(value: ir.Value, index: int, result: ir.Type) -> ir.Value:
    return checked_call("ffi.ArrayGetItem", [value, native_constant(index)], result)


def constant(value: Any) -> ir.Value:
    if isinstance(value, (bool, int, float)):
        return native_constant(value)
    if value is None:
        return ffi.constant_none()
    if isinstance(value, str):
        return ffi.constant_raw_str(value)
    if isinstance(value, torch.device):
        return ffi.constant_device(str(value))
    if isinstance(value, torch.dtype):
        dtype = runtime.convert(value)
        i64 = ir.IntegerType.get_signless(64)
        return ffi.constant_dtype(
            [
                ir.IntegerAttr.get(i64, field)
                for field in (dtype.type_code, dtype.bits, dtype.lanes)
            ]
        )
    if isinstance(value, (tuple, list)):
        return array([constant(element) for element in value])
    if isinstance(value, torch.Tensor):
        # Lifted literals reuse the existing owned runtime materialization.
        cpu = value.detach().cpu().contiguous()
        if cpu.dtype == torch.bfloat16:
            elements = ir.DenseElementsAttr.get(
                cpu.view(torch.int16).numpy(),
                type=ir.RankedTensorType.get(cpu.shape, ir.BF16Type.get()),
            )
        else:
            elements = ir.DenseElementsAttr.get(cpu.numpy())
        return ffi.tensor_literal(elements)
    raise NotImplementedError(f"unsupported FX constant: {value!r}")


def binary(operation: Any, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    lhs, rhs = native(lhs), native(rhs)
    keep_boolean = (
        operation
        in (operator.and_, operator.or_, operator.xor, operator.eq, operator.ne)
        and str(lhs.type) == str(rhs.type) == "i1"
    )
    if not keep_boolean:
        lhs = (
            arith.extui(ir.IntegerType.get_signless(64), lhs)
            if str(lhs.type) == "i1"
            else lhs
        )
        rhs = (
            arith.extui(ir.IntegerType.get_signless(64), rhs)
            if str(rhs.type) == "i1"
            else rhs
        )
    floating = str(lhs.type) == "f64" or str(rhs.type) == "f64"
    if floating or operation is operator.truediv:
        lhs = lhs if str(lhs.type) == "f64" else arith.sitofp(ir.F64Type.get(), lhs)
        rhs = rhs if str(rhs.type) == "f64" else arith.sitofp(ir.F64Type.get(), rhs)
        floating = True
    arithmetic = {
        operator.add: (arith.addi, arith.addf),
        operator.sub: (arith.subi, arith.subf),
        operator.mul: (arith.muli, arith.mulf),
        operator.truediv: (None, arith.divf),
        operator.floordiv: (arith.floordivsi, None),
        operator.and_: (arith.andi, None),
        operator.or_: (arith.ori, None),
        operator.xor: (arith.xori, None),
        operator.lshift: (arith.shli, None),
        operator.rshift: (arith.shrsi, None),
        min: (arith.minsi, arith.minimumf),
        max: (arith.maxsi, arith.maximumf),
    }
    if operation is operator.mod:
        if floating:
            raise NotImplementedError("floating-point Python remainder")
        quotient = arith.floordivsi(lhs, rhs)
        return arith.subi(lhs, arith.muli(quotient, rhs))
    predicates = {
        operator.eq: (arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ),
        operator.ne: (arith.CmpIPredicate.ne, arith.CmpFPredicate.UNE),
        operator.lt: (arith.CmpIPredicate.slt, arith.CmpFPredicate.OLT),
        operator.le: (arith.CmpIPredicate.sle, arith.CmpFPredicate.OLE),
        operator.gt: (arith.CmpIPredicate.sgt, arith.CmpFPredicate.OGT),
        operator.ge: (arith.CmpIPredicate.sge, arith.CmpFPredicate.OGE),
    }
    if operation in predicates:
        integer, float_ = predicates[operation]
        return (
            arith.cmpf(float_, lhs, rhs) if floating else arith.cmpi(integer, lhs, rhs)
        )
    integer, float_ = arithmetic[operation]
    builder = float_ if floating else integer
    if builder is None:
        raise NotImplementedError(f"unsupported scalar operation: {operation}")
    return builder(lhs, rhs)


def equal(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    numeric = ("i1", "i64", "f64", "!tvm_ffi.bool", "!tvm_ffi.int", "!tvm_ffi.float")
    if str(lhs.type) in numeric and str(rhs.type) in numeric:
        return binary(operator.eq, lhs, rhs)
    return ffi.eq(box(lhs), box(rhs))
