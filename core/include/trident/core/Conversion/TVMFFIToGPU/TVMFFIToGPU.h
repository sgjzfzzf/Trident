//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TVMFFITOGPU_TVMFFITOGPU_H_
#define TRIDENT_CORE_CONVERSION_TVMFFITOGPU_TVMFFITOGPU_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/DialectConversion.h>

namespace trident::conversion {

#define GEN_PASS_DECL_CONVERTTVMFFITOGPU
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTVMFFITOGPU
#include "trident/core/Conversion/Passes.h.inc"

void populateTVMFFIToGPUConversionPatterns(mlir::ConversionTarget &target,
                                           mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &typeConverter);

void registerConvertTVMFFIToGPUPass();
} // namespace trident::conversion

#endif // TRIDENT_CORE_CONVERSION_TVMFFITOGPU_TVMFFITOGPU_H_
