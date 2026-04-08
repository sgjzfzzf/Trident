#ifndef LIBTRITON_CORE_C_TVMFFITYPES_H
#define LIBTRITON_CORE_C_TVMFFITYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED bool libtritonCoreTypeIsATVMFFIAnyType(MlirType type);
MLIR_CAPI_EXPORTED bool
libtritonCoreTypeIsATVMFFIObjectHandleType(MlirType type);

MLIR_CAPI_EXPORTED MlirType libtritonCoreTVMFFIAnyTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirType
libtritonCoreTVMFFIObjectHandleTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED MlirTypeID libtritonCoreTVMFFIAnyTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirTypeID
libtritonCoreTVMFFIObjectHandleTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // LIBTRITON_CORE_C_TVMFFITYPES_H
