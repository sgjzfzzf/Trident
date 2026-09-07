//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/IR/TorchExtAttrs.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include <cstdint>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

#include "trident/core/Dialect/TorchExt/IR/TorchExtInterfaces.cpp.inc"

namespace trident::torchext {

namespace {

// Recursive traversal mirrors the nested tuple attribute structure.
// NOLINTBEGIN(misc-no-recursion)
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

mlir::Value materializeConstantValue(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::Attribute value) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Value>(value)
      .Case<mlir::StringAttr>([&](mlir::StringAttr string) -> mlir::Value {
        return mlir::torch::Torch::ConstantStrOp::create(builder, loc, string);
      })
      .Case<mlir::ArrayAttr>([&](mlir::ArrayAttr array) -> mlir::Value {
        llvm::SmallVector<mlir::Value> const elements = llvm::map_to_vector(
            array, [&](mlir::Attribute element) -> mlir::Value {
              return materializeConstantValue(builder, loc, element);
            });
        return mlir::torch::Torch::PrimTupleConstructOp::create(
            builder, loc, getConstantValueType(array, builder.getContext()),
            elements);
      })
      .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr integer) -> mlir::Value {
        if (integer.getType().isInteger(1)) {
          return mlir::torch::Torch::ConstantBoolOp::create(
              builder, loc, integer.getValue().getBoolValue());
        }
        return mlir::torch::Torch::ConstantIntOp::create(builder, loc, integer);
      })
      .Case<mlir::FloatAttr>([&](mlir::FloatAttr floating) -> mlir::Value {
        return mlir::torch::Torch::ConstantFloatOp::create(builder, loc,
                                                           floating);
      });
}
// NOLINTEND(misc-no-recursion)

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
  mlir::Value const expected =
      materializeConstantValue(builder, loc, getValue());
  return EqOp::create(builder, loc, operand, expected);
}

mlir::Type ConstantSpecializationAttr::getTargetType() const {
  return getConstantValueType(getValue(), getContext());
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
