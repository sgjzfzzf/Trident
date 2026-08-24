//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_GENERALIZE_ATEN_OPS_GENERALIZE_ATEN_OPS_H_
#define TRIDENT_CORE_CONVERSION_GENERALIZE_ATEN_OPS_GENERALIZE_ATEN_OPS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

namespace trident::torch {

#define GEN_PASS_DECL_GENERALIZEATENOPS
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_GENERALIZEATENOPS
#include "trident/core/Conversion/Passes.h.inc"

} // namespace trident::torch

#endif // TRIDENT_CORE_CONVERSION_GENERALIZE_ATEN_OPS_GENERALIZE_ATEN_OPS_H_
