//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHEXTTOGPU_TORCHEXTTOGPU_H_
#define TRIDENT_CORE_CONVERSION_TORCHEXTTOGPU_TORCHEXTTOGPU_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace trident::torchext {

#define GEN_PASS_DECL_CONVERTTORCHEXTTOGPU
#include "trident-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHEXTTOGPU
#include "trident-core/Conversion/Passes.h.inc"

void populateTorchExtToGPUConversionPatterns(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter);

void registerConvertTorchExtToGPUPass();
} // namespace trident::torchext

#endif // TRIDENT_CORE_CONVERSION_TORCHEXTTOGPU_TORCHEXTTOGPU_H_
