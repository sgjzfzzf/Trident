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

#define GEN_PASS_DEF_ELIMINATEREFCOUNTPAIRS
#define GEN_PASS_REGISTRATION_ELIMINATEREFCOUNTPAIRS
#include "trident/core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

class EliminateRefCountPairPattern
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
      auto decRef = llvm::dyn_cast<torchext::ObjectDecRefOp>(operation);
      if (!decRef || decRef.getObject() != object)
        continue;

      rewriter.eraseOp(incRef);
      rewriter.eraseOp(decRef);
      return mlir::success();
    }

    return mlir::failure();
  }
};

class EliminateRefCountPairsPass
    : public impl::EliminateRefCountPairsBase<EliminateRefCountPairsPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<EliminateRefCountPairPattern>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace trident::torch
