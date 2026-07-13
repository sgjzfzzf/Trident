//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_H_
#define TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#define GET_OP_CLASSES
#include "trident/core/Dialect/TorchExt/IR/TorchExt.h.inc"

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_H_
