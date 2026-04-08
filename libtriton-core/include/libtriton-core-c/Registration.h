#ifndef LIBTRITON_CORE_C_REGISTRATION_H
#define LIBTRITON_CORE_C_REGISTRATION_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void libtritonCoreRegisterAllDialects(MlirContext context);
MLIR_CAPI_EXPORTED void libtritonCoreRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // LIBTRITON_CORE_C_REGISTRATION_H
