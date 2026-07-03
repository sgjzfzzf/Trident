//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
#define TRIDENT_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace trident::tvm_ffi {

#define GEN_PASS_DECL_CONVERTTVMFFITOLLVM
#include "trident-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTVMFFITOLLVM
#include "trident-core/Conversion/Passes.h.inc"

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns);

void registerConvertTVMFFIToLLVMPass();
void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace trident::tvm_ffi

#endif // TRIDENT_CORE_CONVERSION_TVMFFITOLLVM_TVMFFITOLLVM_H_
