//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/SCF/IR/SCFOps.td"
include "torch-mlir/Dialect/Torch/IR/TorchOps.td"

def : Pat<(Torch_PrimIfYieldOp (variadic $operands)),
          (YieldOp (variadic $operands))>;
