//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHEXTTOTVMFFI_TORCHEXTTOTVMFFI_H_
#define TRIDENT_CORE_CONVERSION_TORCHEXTTOTVMFFI_TORCHEXTTOTVMFFI_H_

#include "mlir/Pass/Pass.h"

namespace trident::torchext {
#define GEN_PASS_DECL_CONVERTTORCHEXTTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"
#define GEN_PASS_REGISTRATION_CONVERTTORCHEXTTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"
} // namespace trident::torchext

#endif
