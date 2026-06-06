#ifndef LIBTRITON_CORE_C_DIALECTS_H
#define LIBTRITON_CORE_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AOTInductor, aoti);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvmffi);

#ifdef __cplusplus
}
#endif

#endif // LIBTRITON_CORE_C_DIALECTS_H
