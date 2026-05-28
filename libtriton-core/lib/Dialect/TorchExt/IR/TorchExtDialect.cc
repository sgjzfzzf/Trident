#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TorchExt/IR/TorchExt.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtTypes.cpp.inc"

namespace libtriton::torch_ext {

bool TritonKernelLaunchOp::hasClusterSize() {
  return getClusterSizeX() && getClusterSizeY() && getClusterSizeZ();
}

void TorchExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libtriton-core/Dialect/TorchExt/IR/TorchExt.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtTypes.cpp.inc"
      >();
}

} // namespace libtriton::torch_ext
