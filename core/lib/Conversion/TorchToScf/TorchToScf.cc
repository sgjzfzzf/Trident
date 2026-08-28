//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchToScf/TorchToScf.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <utility>

namespace trident::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOSCF
#include "trident/core/Conversion/Passes.h.inc"

class ConvertPrimIfYieldOp final
    : public mlir::OpRewritePattern<mlir::torch::Torch::PrimIfYieldOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::PrimIfYieldOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, op.getOperands());
    return mlir::success();
  }
};

class ConvertPrimIfOp final
    : public mlir::OpRewritePattern<mlir::torch::Torch::PrimIfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::PrimIfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    trident::torchext::GetOp condition = trident::torchext::GetOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(), op.getCondition());
    mlir::scf::IfOp replacement = mlir::scf::IfOp::create(
        rewriter, op.getLoc(), op.getResultTypes(), condition.getResult(),
        /*withElseRegion=*/true);
    auto inlineRegion = [&rewriter](mlir::Region &source,
                                    mlir::Region &target) {
      rewriter.inlineRegionBefore(source, target, target.begin());
      rewriter.eraseBlock(&target.back());
    };
    inlineRegion(op.getThenRegion(), replacement.getThenRegion());
    inlineRegion(op.getElseRegion(), replacement.getElseRegion());
    rewriter.replaceOp(op, replacement.getResults());
    return mlir::success();
  }
};

class ConvertTorchToScfPass final
    : public impl::ConvertTorchToScfBase<ConvertTorchToScfPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::scf::SCFDialect,
                           mlir::torch::Torch::TorchDialect,
                           trident::torchext::TorchExtDialect>();
    target.addIllegalOp<mlir::torch::Torch::PrimIfOp,
                        mlir::torch::Torch::PrimIfYieldOp>();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertPrimIfOp, ConvertPrimIfYieldOp>(&getContext());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::torch
