//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TORCHTOTVMFFI_TORCHTOTVMFFI_H_
#define TRIDENT_CORE_CONVERSION_TORCHTOTVMFFI_TORCHTOTVMFFI_H_

#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include <mlir/IR/Types.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace trident::conversion {

/// Add the canonical Torch frontend to semantic TVM FFI mappings to a type
/// converter. Identity conversions and materializations remain the caller's
/// responsibility.
void populateTorchToTVMFFITypeConversions(mlir::TypeConverter &typeConverter);

#define GEN_PASS_DECL_CONVERTTORCHTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"
#define GEN_PASS_REGISTRATION_CONVERTTORCHTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"
} // namespace trident::conversion

#endif
