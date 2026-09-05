//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_DECOMPOSESPECIALIZATION_H_
#define TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_DECOMPOSESPECIALIZATION_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::torchext {

#define GEN_PASS_DECL_DECOMPOSESPECIALIZATION
#include "trident/core/Dialect/TorchExt/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_DECOMPOSESPECIALIZATION
#include "trident/core/Dialect/TorchExt/Transforms/Passes.h.inc"

} // namespace trident::torchext

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_DECOMPOSESPECIALIZATION_H_
