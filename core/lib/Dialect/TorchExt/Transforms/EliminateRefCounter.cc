//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"

namespace trident::torch {

#define GEN_PASS_DEF_ELIMINATEREFCOUNTER
#define GEN_PASS_REGISTRATION_ELIMINATEREFCOUNTER
#include "trident/core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

class EliminateRefCounterPattern
    : public mlir::OpRewritePattern<torchext::ObjectIncRefOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(torchext::ObjectIncRefOp incRef,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value object = incRef.getObject();

    // getNextNode() only visits operations in the same block, so the matched
    // order is always IncRef first and DecRef second.
    for (mlir::Operation *operation = incRef->getNextNode(); operation;
         operation = operation->getNextNode()) {
      if (torchext::ObjectDecRefOp decRef =
              llvm::dyn_cast<torchext::ObjectDecRefOp>(operation);
          decRef && decRef.getObject() == object) {
        rewriter.eraseOp(incRef);
        rewriter.eraseOp(decRef);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

class EliminateRefCounterPass
    : public impl::EliminateRefCounterBase<EliminateRefCounterPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<EliminateRefCounterPattern>(&getContext());

    // Disable folding: the greedy driver folds any op with a folder by
    // default. In particular torch.aten.clone (which has a folder that
    // unconditionally returns its self operand when the types match) would be
    // folded here, aliasing the clone result with its operand and unbalancing
    // the reference counts inserted by RAAI. This pass only cancels
    // IncRef/DecRef pairs; it must not rewrite other ops.
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns),
            mlir::GreedyRewriteConfig().enableFolding(false))))
      signalPassFailure();
  }
};

} // namespace

} // namespace trident::torch
