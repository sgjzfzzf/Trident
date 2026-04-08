#ifndef LIBTRITON_CORE_C_DLPACKOPS_H
#define LIBTRITON_CORE_C_DLPACKOPS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED bool
libtritonCoreOperationIsADLPackFromMemRefOwnedOp(MlirOperation operation);
MLIR_CAPI_EXPORTED bool
libtritonCoreOperationIsADLPackFromMemRefBorrowedOp(MlirOperation operation);
MLIR_CAPI_EXPORTED bool
libtritonCoreOperationIsADLPackViewOp(MlirOperation operation);
MLIR_CAPI_EXPORTED bool
libtritonCoreOperationIsADLPackToMemRefOp(MlirOperation operation);

MLIR_CAPI_EXPORTED MlirValue
libtritonCoreDLPackFromMemRefOwnedGetInput(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreDLPackFromMemRefOwnedGetOutput(MlirOperation operation);

MLIR_CAPI_EXPORTED MlirValue
libtritonCoreDLPackFromMemRefBorrowedGetInput(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreDLPackFromMemRefBorrowedGetOutput(MlirOperation operation);

MLIR_CAPI_EXPORTED MlirValue
libtritonCoreDLPackViewGetInput(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreDLPackViewGetOutput(MlirOperation operation);

MLIR_CAPI_EXPORTED MlirValue
libtritonCoreDLPackToMemRefGetInput(MlirOperation operation);
MLIR_CAPI_EXPORTED MlirValue
libtritonCoreDLPackToMemRefGetOutput(MlirOperation operation);

#ifdef __cplusplus
}
#endif

#endif // LIBTRITON_CORE_C_DLPACKOPS_H
