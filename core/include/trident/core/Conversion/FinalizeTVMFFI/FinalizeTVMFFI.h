//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_FINALIZETVMFFI_FINALIZETVMFFI_H_
#define TRIDENT_CORE_CONVERSION_FINALIZETVMFFI_FINALIZETVMFFI_H_

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::conversion {

#define GEN_PASS_DECL_FINALIZETVMFFI
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_FINALIZETVMFFI
#include "trident/core/Conversion/Passes.h.inc"

void populateFinalizeTVMFFIPatterns(mlir::RewritePatternSet &patterns);
void registerFinalizeTVMFFIPass();

} // namespace trident::conversion

#endif // TRIDENT_CORE_CONVERSION_FINALIZETVMFFI_FINALIZETVMFFI_H_
