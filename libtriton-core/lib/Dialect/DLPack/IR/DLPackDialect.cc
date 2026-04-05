#include "libtriton_core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackOps.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.h"

#include "libtriton_core/Dialect/DLPack/IR/DLPackDialect.cpp.inc"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "libtriton_core/Dialect/DLPack/IR/DLPack.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.cpp.inc"

namespace libtriton::dlpack {

void DLPackDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libtriton_core/Dialect/DLPack/IR/DLPack.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.cpp.inc"
      >();
}

} // namespace libtriton::dlpack
