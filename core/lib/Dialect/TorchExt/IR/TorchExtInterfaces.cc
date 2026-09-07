//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtAttrs.h"
#include <cstdint>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
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
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

#include "trident/core/Dialect/TorchExt/IR/TorchExtInterfaces.cpp.inc"

namespace trident::torchext {

namespace {

bool isValidConstantValue(mlir::Attribute value) {
  if (mlir::isa<mlir::StringAttr>(value)) {
    return true;
  }
  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(value)) {
    return llvm::all_of(array, isValidConstantValue);
  }
  auto typed = mlir::dyn_cast<mlir::TypedAttr>(value);
  if (!typed) {
    return false;
  }
  mlir::Type const type = typed.getType();
  return type.isInteger(1) || type.isInteger(64) || type.isF64();
}

mlir::Type getConstantValueType(mlir::Attribute value,
                                mlir::MLIRContext *context) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Type>(value)
      .Case<mlir::StringAttr>([&](mlir::StringAttr) -> mlir::Type {
        return mlir::torch::Torch::StringType::get(context);
      })
      .Case<mlir::ArrayAttr>([&](mlir::ArrayAttr array) -> mlir::Type {
        return mlir::torch::Torch::TupleType::get(
            context, llvm::map_to_vector(
                         array, [&](mlir::Attribute element) -> mlir::Type {
                           return getConstantValueType(element, context);
                         }));
      })
      .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr integer) -> mlir::Type {
        if (integer.getType().isInteger(1)) {
          return mlir::torch::Torch::BoolType::get(context);
        }
        return mlir::torch::Torch::IntType::get(context);
      })
      .Case<mlir::FloatAttr>([&](mlir::FloatAttr) -> mlir::Type {
        return mlir::torch::Torch::FloatType::get(context);
      });
}

mlir::Value buildTVMFFIConstant(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Attribute value) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Value>(value)
      .Case<mlir::StringAttr>([&](mlir::StringAttr string) -> mlir::Value {
        return tvm_ffi::ConstantRawStrOp::create(builder, loc, string);
      })
      .Case<mlir::ArrayAttr>([&](mlir::ArrayAttr array) -> mlir::Value {
        llvm::SmallVector<mlir::Value> elements;
        elements.reserve(array.size());
        for (mlir::Attribute element : array) {
          elements.push_back(buildTVMFFIConstant(builder, loc, element));
        }
        return tvm_ffi::ArrayCreateOp::create(builder, loc, elements);
      })
      .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr integer) -> mlir::Value {
        if (integer.getType().isInteger(1)) {
          return tvm_ffi::ConstantBoolOp::create(builder, loc, integer);
        }
        return tvm_ffi::ConstantIntOp::create(builder, loc, integer);
      })
      .Case<mlir::FloatAttr>([&](mlir::FloatAttr value) -> mlir::Value {
        return tvm_ffi::ConstantFloatOp::create(builder, loc, value);
      });
}

} // namespace

mlir::LogicalResult ConstantSpecializationAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Attribute value) {
  if (isValidConstantValue(value)) {
    return mlir::success();
  }
  return emitError() << "constant specialization requires an i1, i64, f64, "
                        "string, or tuple value";
}

mlir::Value ConstantSpecializationAttr::buildCheck(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::Value operand) const {
  if (mlir::isa<mlir::ArrayAttr>(getValue())) {
    mlir::Value const expected = buildTVMFFIConstant(builder, loc, getValue());
    return tvm_ffi::EqOp::create(builder, loc, operand, expected);
  }
  if (mlir::isa<mlir::StringAttr>(getValue())) {
    return {};
  }
  auto value = mlir::cast<mlir::TypedAttr>(getValue());
  mlir::Value const expected =
      mlir::LLVM::ConstantOp::create(builder, loc, getTargetType(), value);
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
  if (mlir::isa<mlir::ArrayAttr>(getValue())) {
    return getConstantValueType(getValue(), getContext());
  }
  if (mlir::isa<mlir::StringAttr>(getValue())) {
    return mlir::torch::Torch::StringType::get(getContext());
  }
  return mlir::cast<mlir::TypedAttr>(getValue()).getType();
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
