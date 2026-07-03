//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-core-c/Dialects.h"
#include "mlir/CAPI/Registration.h"
#include "trident-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident-core/Dialect/TorchExt/IR/TorchExtDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TorchExt, torchext,
                                      trident::torchext::TorchExtDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvmffi,
                                      trident::tvm_ffi::TVMFFIDialect)
