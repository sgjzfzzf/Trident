//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchToCf/TorchToCf.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <utility>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTORCHTOCF
#include "trident/core/Conversion/Passes.h.inc"

class ConvertRuntimeAssertOp final
    : public mlir::OpRewritePattern<mlir::torch::Torch::RuntimeAssertOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::RuntimeAssertOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Value cond = op.getCondition();
    mlir::IntegerType const i1Ty = mlir::IntegerType::get(ctx, 1);

    cond = llvm::isa<mlir::LLVM::LLVMStructType>(cond.getType())
               ? mlir::LLVM::TruncOp::create(
                     rewriter, loc, i1Ty,
                     mlir::LLVM::ExtractValueOp::create(
                         rewriter, loc, cond, llvm::ArrayRef<int64_t>{2}))
               : mlir::UnrealizedConversionCastOp::create(rewriter, loc, i1Ty,
                                                          cond)
                     .getResult(0);

    rewriter.replaceOpWithNewOp<mlir::cf::AssertOp>(op, cond, op.getMessage());
    return mlir::success();
  }
};

class ConvertTorchToCfPass final
    : public impl::ConvertTorchToCfBase<ConvertTorchToCfPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    populateTorchToCfConversionPatterns(patterns);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateTorchToCfConversionPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertRuntimeAssertOp>(patterns.getContext());
}

} // namespace trident::conversion
