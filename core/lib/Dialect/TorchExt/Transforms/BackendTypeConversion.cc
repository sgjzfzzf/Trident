//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
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

void trident::torch::setupBackendTypeConversion(
    mlir::ConversionTarget &target, mlir::TypeConverter &typeConverter) {
  setupTorchToTVMFFIAnyConversion(typeConverter);
}
