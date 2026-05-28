#define GEN_PASS_DEF_NORMALIZETORCHEXTOPERANDS
#include "libtriton-core/Dialect/TorchExt/Transforms/TorchExtNormalizeOperands.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::torch_ext {

namespace {

static std::optional<mlir::Value> rewriteKernelOperand(mlir::Value operand) {
  if (mlir::torch::TorchConversion::FromBuiltinTensorOp fromBuiltinOp =
          operand.getDefiningOp<
              mlir::torch::TorchConversion::FromBuiltinTensorOp>()) {
    return fromBuiltinOp.getOperand();
  }
  return std::nullopt;
}

class NormalizeOperandsPattern
    : public mlir::OpRewritePattern<TritonKernelLaunchOp> {
public:
  using mlir::OpRewritePattern<TritonKernelLaunchOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TritonKernelLaunchOp launchOp,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::OperandRange kernelOperands = launchOp.getKernelOperands();
    llvm::SmallVector<mlir::Value> rewrittenKernelOperands;
    rewrittenKernelOperands.reserve(kernelOperands.size());
    bool changed = false;
    for (mlir::Value operand : kernelOperands) {
      std::optional<mlir::Value> rewritten = rewriteKernelOperand(operand);
      changed |= rewritten.has_value();
      rewrittenKernelOperands.push_back(rewritten.value_or(operand));
    }

    if (!changed) {
      return mlir::failure();
    }

    mlir::Type asyncTokenType;
    if (mlir::Value asyncToken = launchOp.getAsyncToken()) {
      asyncTokenType = asyncToken.getType();
    }
    rewriter.replaceOpWithNewOp<TritonKernelLaunchOp>(
        launchOp, asyncTokenType, launchOp.getAsyncDependencies(),
        launchOp.getKernelAttr(), launchOp.getGridSizeX(),
        launchOp.getGridSizeY(), launchOp.getGridSizeZ(),
        launchOp.getBlockSizeX(), launchOp.getBlockSizeY(),
        launchOp.getBlockSizeZ(), launchOp.getClusterSizeX(),
        launchOp.getClusterSizeY(), launchOp.getClusterSizeZ(),
        launchOp.getDynamicSharedMemorySize(), rewrittenKernelOperands,
        launchOp.getAsyncObject());
    return mlir::success();
  }
};

class NormalizeTorchExtOperandsPass
    : public impl::NormalizeTorchExtOperandsBase<
          NormalizeTorchExtOperandsPass> {
public:
  void runOnOperation() final {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<NormalizeOperandsPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace libtriton::torch_ext
