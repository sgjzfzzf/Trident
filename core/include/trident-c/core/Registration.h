//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_C_REGISTRATION_H
#define TRIDENT_CORE_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void tridentCoreRegisterAllDialects(MlirContext context);
MLIR_CAPI_EXPORTED void tridentCoreRegisterAllPasses(void);

/// Return the semantic TVM FFI type corresponding to a Torch frontend type.
/// Types without a specialized representation use !tvm_ffi.any.
MLIR_CAPI_EXPORTED MlirType
tridentCoreConvertTorchTypeToTVMFFIType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // TRIDENT_CORE_C_REGISTRATION_H
