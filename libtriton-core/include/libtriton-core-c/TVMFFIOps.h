#ifndef LIBTRITON_CORE_C_TVMFFIOPS_H
#define LIBTRITON_CORE_C_TVMFFIOPS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED bool
libtritonCoreOperationIsATVMFFIToOp(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreTVMFFIToGetInput(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreTVMFFIToGetOutput(MlirOperation operation);

#ifdef __cplusplus
}
#endif

#endif // LIBTRITON_CORE_C_TVMFFIOPS_H
