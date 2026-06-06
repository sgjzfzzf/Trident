// AOTInductorDialect.cc - AOTInductor Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in AOTInductor.td / AOTInductorTypes.td via
// ODS-generated .cpp.inc files.

#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorDialect.h"
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorDialect.cpp.inc"
#define GET_OP_CLASSES
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductor.cpp.inc"

namespace libtriton::aoti {

void AOTInductorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductor.cpp.inc"
      >();
}

} // namespace libtriton::aoti
