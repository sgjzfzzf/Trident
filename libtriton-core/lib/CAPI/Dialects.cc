#include "libtriton-core-c/Dialects.h"

#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TorchExt, torchext,
                                      libtriton::torchext::TorchExtDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvmffi,
                                      libtriton::tvm_ffi::TVMFFIDialect)
