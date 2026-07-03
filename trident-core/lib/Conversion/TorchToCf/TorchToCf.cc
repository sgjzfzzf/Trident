//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-core/Conversion/TorchToCf/TorchToCf.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "trident-core/Conversion/Utils/Type.h"
#include "trident-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

namespace trident::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOCF
#include "trident-core/Conversion/Passes.h.inc"

namespace {

class ConvertRuntimeAssertOp
    : public mlir::OpRewritePattern<mlir::torch::Torch::RuntimeAssertOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::RuntimeAssertOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Value cond = op.getCondition();
    mlir::IntegerType i1Ty = mlir::IntegerType::get(ctx, 1);

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

class ConvertTorchToCfPass
    : public impl::ConvertTorchToCfBase<ConvertTorchToCfPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    setupBackendTypeConversion(target, typeConverter);
    populateTorchToCfConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTorchToCfConversionPatterns(mlir::ConversionTarget &target,
                                         mlir::TypeConverter &typeConverter,
                                         mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertRuntimeAssertOp>(patterns.getContext());
  target.addLegalDialect<mlir::cf::ControlFlowDialect>();
  target.addIllegalOp<mlir::torch::Torch::RuntimeAssertOp>();
}

} // namespace trident::torch
