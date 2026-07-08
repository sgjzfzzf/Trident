//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// TorchExtDialect.cc - TorchExt Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in TorchExt.td / TorchExtTypes.td via
// ODS-generated .cpp.inc files.

#include "trident-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "trident-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "llvm/ADT/TypeSwitch.h"

#include "trident-core/Dialect/TorchExt/IR/TorchExtDialect.cpp.inc"
#define GET_OP_CLASSES
#include "trident-core/Dialect/TorchExt/IR/TorchExt.cpp.inc"

namespace trident::torchext {

void TorchExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "trident-core/Dialect/TorchExt/IR/TorchExt.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CastOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult CastOp::verify() {
  mlir::Type operandType = getOperand().getType();
  mlir::Type resultType = getResult().getType();

  if ((llvm::isa<mlir::torch::Torch::IntType>(operandType) &&
       llvm::isa<mlir::IntegerType>(resultType)) ||
      (llvm::isa<mlir::torch::Torch::FloatType>(operandType) &&
       llvm::isa<mlir::FloatType>(resultType))) {
    return mlir::success();
  } else {
    return emitOpError("unsupported cast from ")
           << operandType << " to " << resultType;
  }
}

} // namespace trident::torchext
