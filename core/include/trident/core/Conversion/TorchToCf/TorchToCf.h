//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHTOCF_TORCHTOCF_H_
#define TRIDENT_CORE_CONVERSION_TORCHTOCF_TORCHTOCF_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace trident::torch {

#define GEN_PASS_DECL_CONVERTTORCHTOCF
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTORCHTOCF
#include "trident/core/Conversion/Passes.h.inc"

void populateTorchToCfConversionPatterns(mlir::ConversionTarget &target,
                                         mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns);

void registerConvertTorchToCfPass();

} // namespace trident::torch

#endif // TRIDENT_CORE_CONVERSION_TORCHTOCF_TORCHTOCF_H_
