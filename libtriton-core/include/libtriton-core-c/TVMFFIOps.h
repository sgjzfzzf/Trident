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

MLIR_CAPI_EXPORTED bool
libtritonCoreOperationIsATVMFFIStoreOp(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreTVMFFIStoreGetValue(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreTVMFFIStoreGetPtr(MlirOperation operation);

MLIR_CAPI_EXPORTED bool
libtritonCoreOperationIsATVMFFIFunctionCallOp(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreTVMFFIFunctionCallGetFunc(MlirOperation operation);
MLIR_CAPI_EXPORTED int32_t
libtritonCoreTVMFFIFunctionCallGetNumArgs(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreTVMFFIFunctionCallGetArg(MlirOperation operation, int32_t index);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreTVMFFIFunctionCallGetResult(MlirOperation operation);

#ifdef __cplusplus
}
#endif

#endif // LIBTRITON_CORE_C_TVMFFIOPS_H
