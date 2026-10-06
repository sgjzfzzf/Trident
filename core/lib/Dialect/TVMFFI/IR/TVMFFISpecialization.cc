//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIAttrs.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace trident::tvm_ffi {

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
        return tvm_ffi::RawStrType::get(context);
      })
      .Case<mlir::ArrayAttr>([&](mlir::ArrayAttr) -> mlir::Type {
        return tvm_ffi::ArrayType::get(context);
      })
      .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr integer) -> mlir::Type {
        if (integer.getType().isInteger(1)) {
          return tvm_ffi::BoolType::get(context);
        }
        return tvm_ffi::IntType::get(context);
      })
      .Case<mlir::FloatAttr>([&](mlir::FloatAttr) -> mlir::Type {
        return tvm_ffi::FloatType::get(context);
      });
}

mlir::Value buildConstantValue(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Attribute value) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::Value>(value)
      .Case<mlir::StringAttr>([&](mlir::StringAttr string) -> mlir::Value {
        return tvm_ffi::ConstantRawStrOp::create(builder, loc, string);
      })
      .Case<mlir::ArrayAttr>([&](mlir::ArrayAttr array) -> mlir::Value {
        llvm::SmallVector<mlir::Value> const elements = llvm::map_to_vector(
            array, [&](mlir::Attribute element) -> mlir::Value {
              return buildConstantValue(builder, loc, element);
            });
        tvm_ffi::FunctionGetGlobalOp handle =
            tvm_ffi::FunctionGetGlobalOp::create(
                builder, loc, tvm_ffi::FunctionType::get(builder.getContext()),
                builder.getI1Type(), "ffi.Array");
        mlir::cf::AssertOp::create(
            builder, loc, handle.getSuccess(),
            "TVMFFIFunctionGetGlobal failed for ffi.Array");
        mlir::Type const pointerType =
            mlir::LLVM::LLVMPointerType::get(builder.getContext());
        mlir::Value const pointer =
            mlir::UnrealizedConversionCastOp::create(builder, loc, pointerType,
                                                     handle.getResult())
                .getResult(0);
        mlir::Value const null =
            mlir::LLVM::ZeroOp::create(builder, loc, pointerType);
        mlir::Value const nonNull = mlir::LLVM::ICmpOp::create(
            builder, loc, mlir::LLVM::ICmpPredicate::ne, pointer, null);
        mlir::cf::AssertOp::create(
            builder, loc, nonNull,
            "TVMFFIFunctionGetGlobal returned null for ffi.Array");
        tvm_ffi::FunctionCallOp call = tvm_ffi::FunctionCallOp::create(
            builder, loc, tvm_ffi::ArrayType::get(builder.getContext()),
            builder.getI1Type(), handle.getResult(), elements);
        mlir::cf::AssertOp::create(builder, loc, call.getSuccess(),
                                   "TVMFFIFunctionCall failed for ffi.Array");
        return call.getResult();
      })
      .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr integer) -> mlir::Value {
        if (integer.getType().isInteger(1)) {
          return tvm_ffi::ConstantBoolOp::create(
              builder, loc, integer.getValue().getBoolValue());
        }
        return tvm_ffi::ConstantIntOp::create(builder, loc, integer);
      })
      .Case<mlir::FloatAttr>([&](mlir::FloatAttr floating) -> mlir::Value {
        return tvm_ffi::ConstantFloatOp::create(builder, loc, floating);
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
  mlir::Value const expected = buildConstantValue(builder, loc, getValue());
  return tvm_ffi::EqOp::create(builder, loc, operand, expected);
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

} // namespace trident::tvm_ffi
