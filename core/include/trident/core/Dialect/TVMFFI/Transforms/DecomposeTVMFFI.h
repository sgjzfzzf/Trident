//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_DECOMPOSETVMFFI_H_
#define TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_DECOMPOSETVMFFI_H_

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::tvm_ffi {

#define GEN_PASS_DECL_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

} // namespace trident::tvm_ffi

#endif // TRIDENT_CORE_DIALECT_TVMFFI_TRANSFORMS_DECOMPOSETVMFFI_H_
