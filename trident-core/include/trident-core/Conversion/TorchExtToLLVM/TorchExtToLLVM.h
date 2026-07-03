//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_
#define TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace trident::torchext {

#define GEN_PASS_DECL_CONVERTTORCHEXTTOLLVM
#include "trident-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHEXTTOLLVM
#include "trident-core/Conversion/Passes.h.inc"

void populateTorchExtToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTorchExtToLLVMPass();
void registerConvertTorchExtToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace trident::torchext

#endif // TRIDENT_CORE_CONVERSION_TORCHEXTTOLLVM_TORCHEXTTOLLVM_H_
