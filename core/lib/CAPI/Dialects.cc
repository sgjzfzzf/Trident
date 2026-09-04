//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/Dialects.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include <mlir/CAPI/Registration.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TorchExt, torchext,
                                      trident::torchext::TorchExtDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvmffi,
                                      trident::tvm_ffi::TVMFFIDialect)
