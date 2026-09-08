# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from .. import ir
from ._torch_c_ops_gen import *
from .torch import (
    TorchBoolType,
    TorchFloatType,
    TorchGeneratorType,
    TorchIntType,
)


def from_f64(operand, *, loc=None, ip=None):
    return FromF64Op(
        operand,
        results=[TorchFloatType.get(operand.context)],
        loc=loc,
        ip=ip,
    ).result


def from_i1(operand, *, loc=None, ip=None):
    return FromI1Op(
        operand,
        results=[TorchBoolType.get(operand.context)],
        loc=loc,
        ip=ip,
    ).result


def from_i64(operand, *, loc=None, ip=None):
    return FromI64Op(
        operand,
        results=[TorchIntType.get(operand.context)],
        loc=loc,
        ip=ip,
    ).result


def generator_to_i64(operand, *, loc=None, ip=None):
    return GeneratorToI64Op(
        operand,
        results=[ir.IntegerType.get_signless(64, operand.context)],
        loc=loc,
        ip=ip,
    ).result


def get_next_seed(*, loc=None, ip=None):
    return GetNextSeedOp(
        results=[ir.IntegerType.get_signless(64)],
        loc=loc,
        ip=ip,
    ).result


def i64_to_generator(operand, *, loc=None, ip=None):
    return I64ToGeneratorOp(
        operand,
        results=[TorchGeneratorType.get(operand.context)],
        loc=loc,
        ip=ip,
    ).result


def to_f64(operand, *, loc=None, ip=None):
    return ToF64Op(
        operand,
        results=[ir.F64Type.get(operand.context)],
        loc=loc,
        ip=ip,
    ).result


def to_i1(operand, *, loc=None, ip=None):
    return ToI1Op(
        operand,
        results=[ir.IntegerType.get_signless(1, operand.context)],
        loc=loc,
        ip=ip,
    ).result


def to_i64(operand, *, loc=None, ip=None):
    return ToI64Op(
        operand,
        results=[ir.IntegerType.get_signless(64, operand.context)],
        loc=loc,
        ip=ip,
    ).result
