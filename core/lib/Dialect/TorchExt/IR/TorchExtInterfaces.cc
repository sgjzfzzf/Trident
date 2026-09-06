//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/IR/TorchExtAttrs.h"
#include <cstdint>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include "trident/core/Dialect/TorchExt/IR/TorchExtInterfaces.cpp.inc"

namespace trident::torchext {

mlir::LogicalResult ConstantSpecializationAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::TypedAttr value) {
  mlir::Type const type = value.getType();
  if (!type.isInteger(1) && !type.isInteger(64) && !type.isF64()) {
    return emitError()
           << "constant specialization requires an i1, i64, or f64 value";
  }
  return mlir::success();
}

mlir::Value ConstantSpecializationAttr::buildCheck(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::Value operand) const {
  mlir::Value const expected =
      mlir::LLVM::ConstantOp::create(builder, loc, getTargetType(), getValue());
  if (getTargetType().isF64()) {
    mlir::Type const bitsType = builder.getI64Type();
    operand = mlir::LLVM::BitcastOp::create(builder, loc, bitsType, operand);
    mlir::Value const expectedBits =
        mlir::LLVM::BitcastOp::create(builder, loc, bitsType, expected);
    return mlir::LLVM::ICmpOp::create(
        builder, loc,
        mlir::LLVM::ICmpPredicate::eq, // NOLINT(misc-include-cleaner)
        operand, expectedBits);
  }
  return mlir::LLVM::ICmpOp::create(
      builder, loc,
      mlir::LLVM::ICmpPredicate::eq, // NOLINT(misc-include-cleaner)
      operand, expected);
}

mlir::Type ConstantSpecializationAttr::getTargetType() const {
  return getValue().getType();
}

mlir::Value VariableSpecializationAttr::buildCheck(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::Value operand) const {
  uint64_t const divisibility = getDivisibility();
  if (divisibility == 1) {
    return {};
  }
  mlir::Value const value =
      llvm::TypeSwitch<mlir::Type, mlir::Value>(operand.getType())
          .Case<mlir::LLVM::LLVMPointerType>(
              [&](mlir::LLVM::LLVMPointerType) -> mlir::Value {
                return mlir::LLVM::PtrToIntOp::create(
                    builder, loc, builder.getI64Type(), operand);
              })
          .Case<mlir::IntegerType>([&](mlir::IntegerType type) -> mlir::Value {
            return type.isInteger(64)
                       ? operand
                       : mlir::LLVM::SExtOp::create(
                             builder, loc, builder.getI64Type(), operand)
                             .getResult();
          })
          .Default([&](mlir::Type) -> mlir::Value {
            return mlir::UnrealizedConversionCastOp::create(
                       builder, loc, builder.getI64Type(), operand)
                .getResult(0);
          });
  mlir::Value const divisor = mlir::LLVM::ConstantOp::create(
      builder, loc, builder.getI64Type(), llvm::APInt(64, divisibility));
  mlir::Value const remainder =
      mlir::LLVM::URemOp::create(builder, loc, value, divisor);
  mlir::Value const zero =
      mlir::LLVM::ConstantOp::create(builder, loc, builder.getI64Type(), 0);
  return mlir::LLVM::ICmpOp::create(
      builder, loc,
      mlir::LLVM::ICmpPredicate::eq, // NOLINT(misc-include-cleaner)
      remainder, zero);
}

mlir::Type VariableSpecializationAttr::getTargetType() const {
  return getKind().getValue();
}

} // namespace trident::torchext
