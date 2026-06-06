//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

//===----------------------------------------------------------------------===//
// mlir::Type conversion setup.
//===----------------------------------------------------------------------===//

/// Convert Torch tensor types (ValueTensorType, NonValueTensorType) to
/// LLVM pointer type.
static void
setupTorchTensorToLLVMPtrConversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](mlir::torch::Torch::BaseTensorType type) -> std::optional<mlir::Type> {
        return mlir::LLVM::LLVMPointerType::get(type.getContext());
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::LLVM::LLVMPointerType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::BaseTensorType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch OptionalType to an LLVM struct with an i1 tag + contained
/// type. The i1 field indicates whether the optional is None (false) or has a
/// value (true).
static void setupTorchOptionalToLLVMStructConversion(
    mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](
          mlir::torch::Torch::OptionalType type) -> std::optional<mlir::Type> {
        mlir::MLIRContext *ctx = type.getContext();
        mlir::Type containedType = type.getContainedType();
        mlir::Type convertedContained =
            typeConverter.convertType(containedType);
        return mlir::LLVM::LLVMStructType::getLiteral(
            ctx, {mlir::IntegerType::get(ctx, 1), convertedContained});
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::LLVM::LLVMStructType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::OptionalType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch ListType to LLVM pointer type.
static void
setupTorchListToLLVMPtrConversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](mlir::torch::Torch::ListType type) -> std::optional<mlir::Type> {
        return mlir::LLVM::LLVMPointerType::get(type.getContext());
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::LLVM::LLVMPointerType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::ListType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch TupleType to an LLVM struct with each contained type
/// converted via the type converter.
static void
setupTorchTupleToLLVMStructConversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&typeConverter](
          mlir::torch::Torch::TupleType type) -> std::optional<mlir::Type> {
        mlir::MLIRContext *ctx = type.getContext();
        llvm::SmallVector<mlir::Type> convertedTypes;
        for (mlir::Type contained : type.getContainedTypes()) {
          mlir::Type converted = typeConverter.convertType(contained);
          if (!converted)
            return std::nullopt;
          convertedTypes.push_back(converted);
        }
        return mlir::LLVM::LLVMStructType::getLiteral(ctx, convertedTypes);
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::LLVM::LLVMStructType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::TupleType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch BoolType to builtin i1.
static void
setupTorchBoolToI1Conversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](mlir::torch::Torch::BoolType type) -> std::optional<mlir::Type> {
        return mlir::IntegerType::get(type.getContext(), 1);
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::IntegerType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::BoolType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch IntType to builtin i64.
static void
setupTorchIntToI64Conversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](mlir::torch::Torch::IntType type) -> std::optional<mlir::Type> {
        return mlir::IntegerType::get(type.getContext(), 64);
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::IntegerType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::IntType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch FloatType to builtin f64.
static void
setupTorchFloatToF64Conversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](mlir::torch::Torch::FloatType type) -> std::optional<mlir::Type> {
        return mlir::Float64Type::get(type.getContext());
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Float64Type type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::FloatType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch DeviceType to an LLVM struct<i32, i32>.
/// The first i32 represents the device type (e.g., CPU=0, CUDA=1),
/// and the second i32 represents the device index.
static void
setupTorchDeviceToLLVMStructConversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](mlir::torch::Torch::DeviceType type) -> std::optional<mlir::Type> {
        mlir::MLIRContext *ctx = type.getContext();
        return mlir::LLVM::LLVMStructType::getLiteral(
            ctx,
            {mlir::IntegerType::get(ctx, 32), mlir::IntegerType::get(ctx, 32)});
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::LLVM::LLVMStructType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::DeviceType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch NoneType to i64 (zero-initialized placeholder).
static void
setupTorchNoneToI64Conversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](mlir::torch::Torch::NoneType type) -> std::optional<mlir::Type> {
        return mlir::IntegerType::get(type.getContext(), 64);
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::IntegerType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::NoneType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

/// Convert Torch StringType to LLVM pointer type.
static void
setupTorchStringToLLVMPtrConversion(mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](mlir::torch::Torch::StringType type) -> std::optional<mlir::Type> {
        return mlir::LLVM::LLVMPointerType::get(type.getContext());
      });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::LLVM::LLVMPointerType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::torch::Torch::StringType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

void libtriton::torch::populateFuncBackendTypeConversionPatterns(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, typeConverter);
  target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
  target.addDynamicallyLegalOp<mlir::func::CallOp>(
      [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

  mlir::cf::populateCFStructuralTypeConversionsAndLegality(typeConverter,
                                                           patterns, target);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  target.addLegalOp<mlir::ModuleOp>();

  target.markUnknownOpDynamicallyLegal([&](mlir::Operation *op) {
    return mlir::isNotBranchOpInterfaceOrReturnLikeOp(op) ||
           mlir::isLegalForBranchOpInterfaceTypeConversionPattern(
               op, typeConverter) ||
           mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });
}

void libtriton::torch::setupBackendTypeConversion(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter) {
  setupTorchTensorToLLVMPtrConversion(typeConverter);
  setupTorchOptionalToLLVMStructConversion(typeConverter);
  setupTorchListToLLVMPtrConversion(typeConverter);
  setupTorchTupleToLLVMStructConversion(typeConverter);
  setupTorchBoolToI1Conversion(typeConverter);
  setupTorchDeviceToLLVMStructConversion(typeConverter);
  setupTorchIntToI64Conversion(typeConverter);
  setupTorchFloatToF64Conversion(typeConverter);
  setupTorchNoneToI64Conversion(typeConverter);
  setupTorchStringToLLVMPtrConversion(typeConverter);
}
