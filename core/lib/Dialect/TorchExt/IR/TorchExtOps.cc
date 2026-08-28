//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"

#include <mlir/IR/TypeRange.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::torchext {

mlir::LogicalResult ConvertOp::verify() {
  if (mlir::isa<DTypeType>(getOperand().getType()) &&
      mlir::isa<mlir::torch::Torch::IntType>(getResult().getType())) {
    return mlir::success();
  }
  return emitOpError("expects !torchext.dtype -> !torch.int");
}

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
  const mlir::Type input = inputs[0];
  const mlir::Type output = outputs[0];

  return ((llvm::isa<mlir::torch::Torch::IntType, trident::tvm_ffi::IntType>(
              input)) &&
          llvm::isa<mlir::IntegerType>(output)) ||
         ((llvm::isa<mlir::torch::Torch::FloatType,
                     trident::tvm_ffi::FloatType>(input)) &&
          llvm::isa<mlir::FloatType>(output));
}

} // namespace trident::torchext
