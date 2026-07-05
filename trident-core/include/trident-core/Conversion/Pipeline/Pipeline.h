//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_PIPELINE_PIPELINE_H_
#define TRIDENT_CORE_CONVERSION_PIPELINE_PIPELINE_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace trident::torch {

#define GEN_PASS_DECL_TRIDENTLOWERINGPIPELINE
#include "trident-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_TRIDENTLOWERINGPIPELINE
#include "trident-core/Conversion/Passes.h.inc"

} // namespace trident::torch

#endif // TRIDENT_CORE_CONVERSION_PIPELINE_PIPELINE_H_
