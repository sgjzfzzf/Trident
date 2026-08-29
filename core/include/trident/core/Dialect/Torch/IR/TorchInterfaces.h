//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCH_IR_TORCHINTERFACES_H_
#define TRIDENT_CORE_DIALECT_TORCH_IR_TORCHINTERFACES_H_

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Types.h>

#include "trident/core/Dialect/Torch/IR/TorchTypeInterfaces.h.inc"

namespace trident::torch {

void registerTorchToTVMFFITypeInterfaces(mlir::DialectRegistry &registry);

} // namespace trident::torch

#endif // TRIDENT_CORE_DIALECT_TORCH_IR_TORCHINTERFACES_H_
