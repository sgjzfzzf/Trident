//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/Transforms/DecomposeTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace trident::tvm_ffi {

#define GEN_PASS_DEF_DECOMPOSETVMFFI
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

class DecomposeArrayCreateOp : public mlir::OpRewritePattern<ArrayCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCreateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    FunctionGetGlobalOp getGlobal = FunctionGetGlobalOp::create(
        rewriter, op.getLoc(), FunctionType::get(rewriter.getContext()),
        "ffi.Array");
    rewriter.replaceOpWithNewOp<FunctionCallOp>(
        op, op.getResult().getType(), getGlobal.getResult(), op.getElements());
    return mlir::success();
  }
};

class DecomposeArrayGetItemOp : public mlir::OpRewritePattern<ArrayGetItemOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayGetItemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    FunctionGetGlobalOp getGlobal = FunctionGetGlobalOp::create(
        rewriter, op.getLoc(), FunctionType::get(rewriter.getContext()),
        "ffi.ArrayGetItem");
    const llvm::SmallVector<mlir::Value> arguments{op.getArray(),
                                                   op.getIndex()};
    rewriter.replaceOpWithNewOp<FunctionCallOp>(
        op, op.getResult().getType(), getGlobal.getResult(), arguments);
    return mlir::success();
  }
};

class DecomposeTVMFFIPass
    : public impl::DecomposeTVMFFIBase<DecomposeTVMFFIPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeArrayCreateOp, DecomposeArrayGetItemOp>(
        &getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::tvm_ffi
