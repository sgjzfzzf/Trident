//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHCONVERSIONTOLLVM_TORCHCONVERSIONTOLLVM_H_
#define TRIDENT_CORE_CONVERSION_TORCHCONVERSIONTOLLVM_TORCHCONVERSIONTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace trident::torch {

#define GEN_PASS_DECL_CONVERTTORCHCONVERSIONTOLLVM
#include "trident-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHCONVERSIONTOLLVM
#include "trident-core/Conversion/Passes.h.inc"

void populateTorchConversionToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTorchConversionToLLVMPass();
void registerConvertTorchConversionToLLVMInterface(
    mlir::DialectRegistry &registry);

} // namespace trident::torch

#endif // TRIDENT_CORE_CONVERSION_TORCHCONVERSIONTOLLVM_TORCHCONVERSIONTOLLVM_H_
