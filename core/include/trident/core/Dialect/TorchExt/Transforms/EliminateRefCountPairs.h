//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_ELIMINATEREFCOUNTPAIRS_H_
#define TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_ELIMINATEREFCOUNTPAIRS_H_

#include "mlir/Pass/Pass.h"

namespace trident::torch {

#define GEN_PASS_DECL_ELIMINATEREFCOUNTPAIRS
#include "trident-core/Dialect/TorchExt/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_ELIMINATEREFCOUNTPAIRS
#include "trident-core/Dialect/TorchExt/Transforms/Passes.h.inc"

} // namespace trident::torch

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_ELIMINATEREFCOUNTPAIRS_H_
