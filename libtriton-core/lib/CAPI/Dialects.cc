#include "libtriton-core-c/Dialects.h"

#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DLPack, dlpack,
                                      libtriton::dlpack::DLPackDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TorchExt, torch_ext,
                                      libtriton::torch_ext::TorchExtDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TVMFFI, tvm_ffi,
                                      libtriton::tvm_ffi::TVMFFIDialect)
