# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Shared IR builders for graph import, input handling and Dynamo guards.

Scalars stay native while computing and are boxed at FFI boundaries.
Checked calls validate lookup status, function handles and call status.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch
import tvm_ffi as runtime

from trident.core import ir
from trident.core.dialects import arith, builtin, cf, llvm
from trident.core.dialects import tvm_ffi as tvm_ffi_d


def _is_native_scalar(type_: ir.Type) -> bool:
    return isinstance(type_, ir.F64Type) or (
        isinstance(type_, ir.IntegerType)
        and type_.is_signless
        and type_.width in (1, 64)
    )


def _is_numeric_scalar(type_: ir.Type) -> bool:
    return _is_native_scalar(type_) or isinstance(
        type_, (tvm_ffi_d.BoolType, tvm_ffi_d.FloatType, tvm_ffi_d.IntType)
    )


def _is_signless_integer(type_: ir.Type, width: int) -> bool:
    return (
        isinstance(type_, ir.IntegerType) and type_.is_signless and type_.width == width
    )


def _wrap_bool(
    builder: Callable[[ir.Value, ir.Value], ir.Value],
) -> Callable[[ir.Value, ir.Value], ir.Value]:
    def wrapped(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        operands = []
        for value in (lhs, rhs):
            if isinstance(value.type, ir.F64Type):
                value = arith.cmpf(arith.CmpFPredicate.UNE, value, native_constant(0.0))
            elif not _is_signless_integer(value.type, 1):
                value = arith.cmpi(arith.CmpIPredicate.ne, value, native_constant(0))
            operands.append(value)
        return builder(*operands)

    return wrapped


def _wrap_floating(
    builder: Callable[[ir.Value, ir.Value], ir.Value],
) -> Callable[[ir.Value, ir.Value], ir.Value]:
    def wrapped(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        operands = []
        for value in (lhs, rhs):
            if _is_signless_integer(value.type, 1):
                value = arith.extui(ir.IntegerType.get_signless(64), value)
            if not isinstance(value.type, ir.F64Type):
                value = arith.sitofp(ir.F64Type.get(), value)
            operands.append(value)
        return builder(*operands)

    return wrapped


def _wrap_integer(
    builder: Callable[[ir.Value, ir.Value], ir.Value],
) -> Callable[[ir.Value, ir.Value], ir.Value]:
    def wrapped(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        if any(isinstance(value.type, ir.F64Type) for value in (lhs, rhs)):
            raise NotImplementedError("floating-point input to integer operation")
        return builder(
            *(
                arith.extui(ir.IntegerType.get_signless(64), value)
                if _is_signless_integer(value.type, 1)
                else value
                for value in (lhs, rhs)
            )
        )

    return wrapped


# Entries contain boolean, integer and floating-point implementations, in that order.
_BINARY_ARITHMETIC: dict[
    Callable[..., Any],
    tuple[
        Callable[[ir.Value, ir.Value], ir.Value] | None,
        Callable[[ir.Value, ir.Value], ir.Value] | None,
        Callable[[ir.Value, ir.Value], ir.Value] | None,
    ],
] = {
    max: (None, _wrap_integer(arith.maxsi), _wrap_floating(arith.maximumf)),
    min: (None, _wrap_integer(arith.minsi), _wrap_floating(arith.minimumf)),
    operator.add: (None, _wrap_integer(arith.addi), _wrap_floating(arith.addf)),
    operator.and_: (_wrap_bool(arith.andi), _wrap_integer(arith.andi), None),
    operator.eq: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.eq)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OEQ)),
    ),
    operator.floordiv: (None, _wrap_integer(arith.floordivsi), None),
    operator.ge: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.sge)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OGE)),
    ),
    operator.gt: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.sgt)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OGT)),
    ),
    operator.le: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.sle)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OLE)),
    ),
    operator.lshift: (None, _wrap_integer(arith.shli), None),
    operator.lt: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.slt)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.OLT)),
    ),
    operator.mod: (
        None,
        _wrap_integer(
            lambda lhs, rhs: arith.subi(
                lhs, arith.muli(arith.floordivsi(lhs, rhs), rhs)
            )
        ),
        None,
    ),
    operator.mul: (None, _wrap_integer(arith.muli), _wrap_floating(arith.mulf)),
    operator.ne: (
        None,
        _wrap_integer(partial(arith.cmpi, arith.CmpIPredicate.ne)),
        _wrap_floating(partial(arith.cmpf, arith.CmpFPredicate.UNE)),
    ),
    operator.or_: (_wrap_bool(arith.ori), _wrap_integer(arith.ori), None),
    operator.rshift: (None, _wrap_integer(arith.shrsi), None),
    operator.sub: (None, _wrap_integer(arith.subi), _wrap_floating(arith.subf)),
    operator.truediv: (None, None, _wrap_floating(arith.divf)),
    operator.xor: (_wrap_bool(arith.xori), _wrap_integer(arith.xori), None),
}


def array(values: Sequence[ir.Value]) -> ir.Value:
    return checked_call("ffi.Array", values, tvm_ffi_d.ArrayType.get())


def array_get(value: ir.Value, index: int, result: ir.Type) -> ir.Value:
    return checked_call("ffi.ArrayGetItem", [value, native_constant(index)], result)


def binary(operation: Any, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    lhs, rhs = native(lhs), native(rhs)
    boolean, integer, floating = _BINARY_ARITHMETIC[operation]
    if operation is operator.truediv or any(
        isinstance(value.type, ir.F64Type) for value in (lhs, rhs)
    ):
        builder = floating
    elif boolean is not None and all(
        _is_signless_integer(value.type, 1) for value in (lhs, rhs)
    ):
        builder = boolean
    else:
        builder = integer
    if builder is None:
        raise NotImplementedError(f"unsupported scalar operation: {operation}")
    return builder(lhs, rhs)


def box(value: ir.Value) -> ir.Value:
    return tvm_ffi_d.to(value) if _is_native_scalar(value.type) else value


def checked_call(name: str, arguments: Sequence[ir.Value], result: ir.Type) -> ir.Value:
    handle = tvm_ffi_d.FunctionGetGlobalOp(name)
    cf.AssertOp(handle.success, f"TVMFFIFunctionGetGlobal failed for {name}")
    pointer = builtin.unrealized_conversion_cast(
        [llvm.PointerType.get()], [handle.result]
    )
    null = llvm.mlir_zero(llvm.PointerType.get())
    nonnull = llvm.icmp(llvm.ICmpPredicate.ne, pointer, null)
    cf.AssertOp(nonnull, f"TVMFFIFunctionGetGlobal returned null for {name}")
    call = tvm_ffi_d.FunctionCallOp(
        result,
        ir.IntegerType.get_signless(1),
        handle.result,
        [box(argument) for argument in arguments],
    )
    cf.AssertOp(call.success, f"TVMFFIFunctionCall failed for {name}")
    return call.result


def constant(value: Any) -> ir.Value:
    if isinstance(value, (bool, int, float)):
        return native_constant(value)
    if value is None:
        return tvm_ffi_d.constant_none()
    if isinstance(value, str):
        return tvm_ffi_d.constant_raw_str(value)
    if isinstance(value, torch.device):
        return tvm_ffi_d.constant_device(f"{value}")
    if isinstance(value, torch.dtype):
        dtype = runtime.convert(value)
        i64 = ir.IntegerType.get_signless(64)
        return tvm_ffi_d.constant_dtype(
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
        return tvm_ffi_d.tensor_literal(elements)
    raise NotImplementedError(f"unsupported FX constant: {value!r}")


def equal(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    if _is_numeric_scalar(lhs.type) and _is_numeric_scalar(rhs.type):
        return binary(operator.eq, lhs, rhs)
    return tvm_ffi_d.eq(box(lhs), box(rhs))


def ffi_type(name: str) -> ir.Type:
    """Compatibility constructor for a textual TVM FFI type description."""
    return ir.Type.parse(f"!tvm_ffi.{name}")


def is_binary_operation(operation: Any) -> bool:
    """Whether a callable has a native binary scalar implementation."""
    return operation in _BINARY_ARITHMETIC


def native(value: ir.Value) -> ir.Value:
    return (
        tvm_ffi_d.get(value)
        if isinstance(
            value.type, (tvm_ffi_d.BoolType, tvm_ffi_d.IntType, tvm_ffi_d.FloatType)
        )
        else value
    )


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


def value_type(value: Any) -> ir.Type:
    if isinstance(value, torch.Tensor):
        return tvm_ffi_d.TensorType.get()
    if isinstance(value, (bool, torch.SymBool)):
        return tvm_ffi_d.BoolType.get()
    if isinstance(value, (int, torch.SymInt)):
        return tvm_ffi_d.IntType.get()
    if isinstance(value, (float, torch.SymFloat)):
        return tvm_ffi_d.FloatType.get()
    if isinstance(value, str):
        return tvm_ffi_d.UnionType.get(
            [
                tvm_ffi_d.RawStrType.get(),
                tvm_ffi_d.SmallStrType.get(),
                tvm_ffi_d.StrType.get(),
            ]
        )
    if isinstance(value, torch.device):
        return tvm_ffi_d.DeviceType.get()
    if isinstance(value, torch.dtype):
        return tvm_ffi_d.DTypeType.get()
    if isinstance(value, (tuple, list, dict)):
        return tvm_ffi_d.ArrayType.get()
    if value is None:
        return tvm_ffi_d.NoneType.get()
    raise NotImplementedError(f"unsupported FX value type: {type(value)}")
