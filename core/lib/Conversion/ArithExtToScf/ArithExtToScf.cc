//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/ArithExtToScf/ArithExtToScf.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/ArithExt/IR/ArithExtOps.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTARITHEXTTOSCF
#include "trident/core/Conversion/Passes.h.inc"

// The recursive shape mirrors the nested scf.if structure being built.
// NOLINTNEXTLINE(misc-no-recursion)
mlir::Value buildAndThenChain(mlir::RewriterBase &rewriter, mlir::Location loc,
                              llvm::ArrayRef<mlir::Region *> regions,
                              mlir::Value condition) {
  if (regions.empty()) {
    return condition;
  } else {
    const mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::scf::IfOp ifOp = mlir::scf::IfOp::create(
        rewriter, loc, mlir::TypeRange{rewriter.getI1Type()}, condition,
        /*addThenBlock=*/true, /*addElseBlock=*/true);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    mlir::IRMapping mapping;
    for (mlir::Operation &operation : regions.front()->front()) {
      if (arithext::AndThenYieldOp yield =
              llvm::dyn_cast<arithext::AndThenYieldOp>(&operation)) {
        mlir::Value thenValue = mapping.lookupOrDefault(yield.getValue());
        thenValue =
            buildAndThenChain(rewriter, loc, regions.drop_front(), thenValue);
        mlir::scf::YieldOp::create(rewriter, loc, thenValue);
      } else {
        rewriter.clone(operation, mapping);
      }
    }
    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    const mlir::Value falseValue =
        mlir::arith::ConstantIntOp::create(rewriter, loc, false, 1);
    mlir::scf::YieldOp::create(rewriter, loc, falseValue);
    return ifOp.getResult(0);
  }
}

class ConvertAndThenOp final
    : public mlir::OpRewritePattern<arithext::AndThenOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(arithext::AndThenOp op,
                  mlir::PatternRewriter &rewriter) const override {
    const mlir::Value trueValue =
        mlir::arith::ConstantIntOp::create(rewriter, op.getLoc(), true, 1);
    const llvm::SmallVector<mlir::Region *> regions = llvm::map_to_vector(
        op.getRegions(), [](mlir::Region &region) { return &region; });
    const mlir::Value result =
        buildAndThenChain(rewriter, op.getLoc(), regions, trueValue);
    rewriter.replaceOp(op, result);
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
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::conversion
