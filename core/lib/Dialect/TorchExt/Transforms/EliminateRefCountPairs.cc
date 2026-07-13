//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"

namespace trident::torch {

#define GEN_PASS_DEF_ELIMINATEREFCOUNTPAIRS
#define GEN_PASS_REGISTRATION_ELIMINATEREFCOUNTPAIRS
#include "trident/core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

class EliminateRefCountPairsPass
    : public impl::EliminateRefCountPairsBase<EliminateRefCountPairsPass> {
public:
  void runOnOperation() final {
    getOperation()->walk([&](mlir::Block *block) {
      llvm::DenseMap<mlir::Value,
                     llvm::SmallVector<torchext::ObjectIncRefOp, 1>>
          unmatchedIncRefs;
      llvm::SmallVector<mlir::Operation *> operationsToErase;

      // Only inspect operations directly contained in this block. Nested
      // blocks are visited separately, so pairs never cross block boundaries.
      for (mlir::Operation &operation : *block) {
        if (auto incRef =
                llvm::dyn_cast<torchext::ObjectIncRefOp>(&operation)) {
          unmatchedIncRefs[incRef.getObject()].push_back(incRef);
          continue;
        }

        auto decRef = llvm::dyn_cast<torchext::ObjectDecRefOp>(&operation);
        if (!decRef)
          continue;

        auto it = unmatchedIncRefs.find(decRef.getObject());
        if (it == unmatchedIncRefs.end() || it->second.empty())
          continue;

        torchext::ObjectIncRefOp incRef = it->second.pop_back_val();
        operationsToErase.push_back(incRef.getOperation());
        operationsToErase.push_back(decRef.getOperation());
      }

      // Delay erasure until after the scan to keep the block iterator valid.
      for (mlir::Operation *operation : operationsToErase)
        operation->erase();
    });
  }
};

} // namespace

} // namespace trident::torch
