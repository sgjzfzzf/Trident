//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "trident-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident-core/Dialect/TorchExt/IR/TorchExtOps.h"

namespace trident::torch {

#define GEN_PASS_DEF_RAAI
#define GEN_PASS_REGISTRATION_RAAI
#include "trident-core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

class RAAIPass : public impl::RAAIBase<RAAIPass> {
public:
  void runOnOperation() final {
    getOperation().walk([&](mlir::Operation *op) {
      // Skip the module itself — its body is not a meaningful scope
      // for per-block reference counting.
      if (llvm::isa<mlir::ModuleOp>(op)) {
        return;
      }
      for (mlir::Region &region : op->getRegions()) {
        if (!region.hasOneBlock()) {
          continue;
        }
        mlir::Block &block = region.front();

        // --- Collect allocated values ---
        llvm::SmallVector<mlir::Value> allocatedVals;
        block.walk([&](mlir::Operation *op) {
          if (mlir::torch::Torch::PrimListConstructOp listCtor =
                  llvm::dyn_cast<mlir::torch::Torch::PrimListConstructOp>(op)) {
            allocatedVals.push_back(listCtor.getResult());
          } else if (mlir::torch::Torch::ValueTensorLiteralOp literalOp =
                         llvm::dyn_cast<
                             mlir::torch::Torch::ValueTensorLiteralOp>(op)) {
            allocatedVals.push_back(literalOp.getResult());
          } else if (mlir::OperationName opName = op->getName();
                     opName.getDialectNamespace() == "torch" &&
                     opName.getStringRef().starts_with("torch.aten.")) {
            for (mlir::Value result :
                 llvm::make_filter_range(op->getResults(), [](mlir::Value val) {
                   return llvm::isa<mlir::torch::Torch::ValueTensorType,
                                    mlir::torch::Torch::NonValueTensorType,
                                    mlir::torch::Torch::ListType>(
                       val.getType());
                 })) {
              allocatedVals.push_back(result);
            }
          }
        });

        // --- Insert ref-count ops before the terminator ---
        mlir::Operation *terminator = block.getTerminator();
        mlir::OpBuilder builder(terminator);
        mlir::Location loc = terminator->getLoc();
        mlir::ValueRange yieldedVals = terminator->getOperands();

        for (mlir::Value val : yieldedVals) {
          torchext::ObjectIncRefOp::create(builder, loc, val);
        }

        for (mlir::Value val : allocatedVals) {
          torchext::ObjectDecRefOp::create(builder, loc, val);
        }
      }
    });
  }
};

} // namespace

} // namespace trident::torch
