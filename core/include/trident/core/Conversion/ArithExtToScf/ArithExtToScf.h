//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_ARITHEXT_TO_SCF_ARITHEXT_TO_SCF_H_
#define TRIDENT_CORE_CONVERSION_ARITHEXT_TO_SCF_ARITHEXT_TO_SCF_H_

#include "trident/core/Dialect/ArithExt/IR/ArithExtDialect.h"
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::conversion {

#define GEN_PASS_DECL_CONVERTARITHEXTTOSCF
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTARITHEXTTOSCF
#include "trident/core/Conversion/Passes.h.inc"

} // namespace trident::conversion

#endif // TRIDENT_CORE_CONVERSION_ARITHEXT_TO_SCF_ARITHEXT_TO_SCF_H_
