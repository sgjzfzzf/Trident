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

#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <llvm/ADT/TypeSwitch.h> // NOLINT(misc-include-cleaner)
#include <llvm/Support/Casting.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h> // NOLINT(misc-include-cleaner)
#include <mlir/Support/LLVM.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.cpp.inc"
#define GET_OP_CLASSES
#include "trident/core/Dialect/TorchExt/IR/TorchExt.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.cpp.inc"

namespace trident::torchext {

mlir::LogicalResult ConvertOp::verify() {
  if (mlir::isa<DTypeType>(getOperand().getType()) &&
      mlir::isa<mlir::torch::Torch::IntType>(getResult().getType())) {
    return mlir::success();
  } else {
    return emitOpError("expects !torchext.dtype -> !torch.int");
  }
}

void TorchExtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/TorchExt/IR/TorchExt.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CastOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult CastOp::verify() {
  return areCastCompatible(getOperand().getType(), getResult().getType())
             ? mlir::success()
             : emitOpError("unsupported cast from ")
                   << getOperand().getType() << " to " << getResult().getType();
}

bool CastOp::areCastCompatible(mlir::TypeRange inputs,
                               mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }
  mlir::Type input = inputs[0];
  mlir::Type output = outputs[0];

  return ((llvm::isa<mlir::torch::Torch::IntType, trident::tvm_ffi::IntType>(
              input)) &&
          llvm::isa<mlir::IntegerType>(output)) ||
         ((llvm::isa<mlir::torch::Torch::FloatType,
                     trident::tvm_ffi::FloatType>(input)) &&
          llvm::isa<mlir::FloatType>(output));
}

} // namespace trident::torchext
