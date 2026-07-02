//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_PIPELINE_PIPELINE_H_
#define LIBTRITON_CORE_CONVERSION_PIPELINE_PIPELINE_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace libtriton::torch {

#define GEN_PASS_DECL_TORCHTOLLVMPIPELINE
#include "libtriton-core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_TORCHTOLLVMPIPELINE
#include "libtriton-core/Conversion/Passes.h.inc"

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_CONVERSION_PIPELINE_PIPELINE_H_
