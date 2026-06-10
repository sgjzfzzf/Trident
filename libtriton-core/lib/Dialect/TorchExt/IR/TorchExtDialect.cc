// TorchExtDialect.cc - TorchExt Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in TorchExt.td / TorchExtTypes.td via
// ODS-generated .cpp.inc files.

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.cpp.inc"
#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TorchExt/IR/TorchExt.cpp.inc"

namespace libtriton::torchext {

void TorchExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libtriton-core/Dialect/TorchExt/IR/TorchExt.cpp.inc"
      >();
}

} // namespace libtriton::torchext
