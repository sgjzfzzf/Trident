//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHTOTVMFFI_TORCHTOTVMFFI_H_
#define TRIDENT_CORE_CONVERSION_TORCHTOTVMFFI_TORCHTOTVMFFI_H_

#include <mlir/Pass/Pass.h>

namespace trident::conversion {

#define GEN_PASS_DECL_CONVERTTORCHTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"
#define GEN_PASS_REGISTRATION_CONVERTTORCHTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"
} // namespace trident::conversion

#endif
