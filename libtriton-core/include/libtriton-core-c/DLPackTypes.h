#ifndef LIBTRITON_CORE_C_DLPACKTYPES_H
#define LIBTRITON_CORE_C_DLPACKTYPES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED bool libtritonCoreTypeIsADLPackDLContextType(MlirType type);
MLIR_CAPI_EXPORTED bool libtritonCoreTypeIsADLPackDLDataTypeType(MlirType type);
MLIR_CAPI_EXPORTED bool libtritonCoreTypeIsADLPackDLTensorType(MlirType type);
MLIR_CAPI_EXPORTED bool
libtritonCoreTypeIsADLPackDLManagedTensorType(MlirType type);

MLIR_CAPI_EXPORTED MlirType
libtritonCoreDLPackDLContextTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirType
libtritonCoreDLPackDLDataTypeTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirType
libtritonCoreDLPackDLTensorTypeGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirType
libtritonCoreDLPackDLManagedTensorTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED MlirTypeID libtritonCoreDLPackDLContextTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirTypeID libtritonCoreDLPackDLDataTypeTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirTypeID libtritonCoreDLPackDLTensorTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirTypeID
libtritonCoreDLPackDLManagedTensorTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // LIBTRITON_CORE_C_DLPACKTYPES_H
