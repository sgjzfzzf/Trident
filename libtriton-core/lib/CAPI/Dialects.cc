#include "libtriton-core-c/Dialects.h"

#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AOTInductor, aoti,
                                      libtriton::aoti::AOTInductorDialect)
