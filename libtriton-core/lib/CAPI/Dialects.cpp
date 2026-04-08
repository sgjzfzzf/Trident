#include "libtriton-core-c/Dialects.h"

#include "libtriton_core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DLPack, dlpack,
                                      libtriton::dlpack::DLPackDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvm_ffi,
                                      libtriton::tvm_ffi::TVMFFIDialect)
