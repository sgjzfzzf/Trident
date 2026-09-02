//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
#define TRIDENT_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/DialectConversion.h>

namespace trident::conversion {

#define GEN_PASS_DECL_CONVERTDLPACKTOLLVM
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTDLPACKTOLLVM
#include "trident/core/Conversion/Passes.h.inc"

void populateDLPackToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertDLPackToLLVMPass();
void registerConvertDLPackToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace trident::conversion

#endif // TRIDENT_CORE_CONVERSION_DLPACKTOLLVM_DLPACKTOLLVM_H_
