// TVMFFIDialect.cc - TVMFFI Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in TVMFFI.td / TVMFFITypes.td via ODS-generated
// .cpp.inc files.

#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"

namespace libtriton::tvm_ffi {

void TVMFFIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"
      >();
}

} // namespace libtriton::tvm_ffi
