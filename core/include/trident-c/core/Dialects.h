//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_C_DIALECTS_H
#define TRIDENT_CORE_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TorchExt, torchext);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvmffi);

#ifdef __cplusplus
}
#endif

#endif // TRIDENT_CORE_C_DIALECTS_H
