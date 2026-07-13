//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "trident/core/Conversion/Utils/Type.h"

//===----------------------------------------------------------------------===//
// mlir::Type conversion setup.
//===----------------------------------------------------------------------===//

/// Convert all Torch dialect types uniformly to TVMFFIAny
/// (!llvm.struct<(i32, i32, i64)>). Materializations use
/// UnrealizedConversionCastOp as placeholders; real pack/unpack is performed
/// by downstream passes (CAPI, TVMFFIToLLVM).
static void
setupTorchToTVMFFIAnyConversion(mlir::TypeConverter &typeConverter) {
  typeConverter.addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
    if (llvm::isa<mlir::torch::Torch::TorchDialect>(type.getDialect())) {
      return trident::conversion::utils::getTVMFFIAnyType(type.getContext());
    } else {
      return std::nullopt;
    }
  });
  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::LLVM::LLVMStructType type,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
         mlir::Location loc) -> mlir::Value {
        return mlir::UnrealizedConversionCastOp::create(
                   builder, loc, mlir::TypeRange(type), inputs)
            .getResult(0);
      });
}

void trident::torch::populateFuncBackendTypeConversionPatterns(
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

void trident::torch::setupBackendTypeConversion(
    mlir::ConversionTarget &target, mlir::TypeConverter &typeConverter) {
  setupTorchToTVMFFIAnyConversion(typeConverter);
}
