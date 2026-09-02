//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/Torch/IR/TorchInterfaces.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>

namespace trident::torchext {

mlir::LogicalResult ConvertOp::verify() {
  if (mlir::isa<DTypeType>(getOperand().getType()) &&
      mlir::isa<mlir::torch::Torch::IntType>(getResult().getType())) {
    return mlir::success();
  }
  return emitOpError("expects !torchext.dtype -> !torch.int");
}

mlir::LogicalResult GetOp::verify() {
  mlir::Type const input = getOperand().getType();
  mlir::Type const output = getResult().getType();
  mlir::Type tvmFFIType = input;
  if (auto torchTypeInterface =
          mlir::dyn_cast<trident::torch::TorchToTVMFFITypeInterface>(input)) {
    tvmFFIType = torchTypeInterface.getTVMFFIType();
  }

  auto nativeTypeInterface =
      mlir::dyn_cast<tvm_ffi::TVMFFINativeTypeInterface>(tvmFFIType);
  if (!nativeTypeInterface) {
    return emitOpError("unsupported get from ") << input << " to " << output;
  }

  mlir::Type const nativeType = nativeTypeInterface.getNativeType();
  if (nativeType == output) {
    return mlir::success();
  }
  return emitOpError("unsupported get from ") << input << " to " << output;
}

} // namespace trident::torchext
