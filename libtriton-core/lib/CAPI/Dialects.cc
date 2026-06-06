#include "libtriton-core-c/Dialects.h"

#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AOTInductor, aoti,
                                      libtriton::aoti::AOTInductorDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvmffi,
                                      libtriton::tvm_ffi::TVMFFIDialect)
