//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/ArithExtToScf/ArithExtToScf.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "trident/core/Dialect/ArithExt/IR/ArithExtOps.h"

namespace trident::arithex {

#define GEN_PASS_DEF_CONVERTARITHEXTTOSCF
#include "trident/core/Conversion/Passes.h.inc"

namespace {

mlir::Value buildAndThenChain(mlir::PatternRewriter &rewriter,
                              mlir::Location loc,
                              mlir::MutableArrayRef<mlir::Region> regions,
                              mlir::Value condition) {
  mlir::scf::IfOp ifOp = mlir::scf::IfOp::create(
      rewriter, loc, mlir::TypeRange{rewriter.getI1Type()}, condition,
      /*addThenBlock=*/true, /*addElseBlock=*/true);

  ifOp.getThenRegion().takeBody(regions.front());
  AndThenYieldOp yield =
      llvm::cast<AndThenYieldOp>(ifOp.thenBlock()->getTerminator());
  rewriter.setInsertionPoint(yield);
  mlir::Value thenValue = yield.getValue();
  mlir::MutableArrayRef<mlir::Region> remainingRegions = regions.drop_front();
  if (!remainingRegions.empty()) {
    thenValue = buildAndThenChain(rewriter, yield.getLoc(), remainingRegions,
                                  thenValue);
  }
  rewriter.setInsertionPoint(yield);
  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(yield, thenValue);
  rewriter.setInsertionPointToEnd(ifOp.elseBlock());
  mlir::Value falseValue =
      mlir::arith::ConstantIntOp::create(rewriter, loc, false, 1);
  mlir::scf::YieldOp::create(rewriter, loc, falseValue);
  return ifOp.getResult(0);
}

class ConvertAndThenOp final : public mlir::OpRewritePattern<AndThenOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AndThenOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getNumRegions() == 0) {
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(op, true, 1);
    } else {
      mlir::Value trueValue =
          mlir::arith::ConstantIntOp::create(rewriter, op.getLoc(), true, 1);
      mlir::MutableArrayRef<mlir::Region> regions = op.getRegions();
      mlir::Value result =
          buildAndThenChain(rewriter, op.getLoc(), regions, trueValue);
      rewriter.replaceOp(op, result);
    }
    return mlir::success();
  }
};

class ConvertArithExtToScfPass
    : public impl::ConvertArithExtToScfBase<ConvertArithExtToScfPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertAndThenOp>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace trident::arithex
