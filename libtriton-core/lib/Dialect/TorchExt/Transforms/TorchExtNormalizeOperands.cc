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

static mlir::Value rewriteKernelOperand(mlir::Value operand) {
  if (mlir::torch::TorchConversion::FromBuiltinTensorOp fromBuiltinOp =
          operand.getDefiningOp<
              mlir::torch::TorchConversion::FromBuiltinTensorOp>()) {
    return fromBuiltinOp.getOperand();
  } else {
    return operand;
  }
}

class NormalizeOperandsPattern
    : public mlir::OpRewritePattern<libtriton::torch_ext::TritonKernelLaunchOp> {
public:
  using mlir::OpRewritePattern<
      libtriton::torch_ext::TritonKernelLaunchOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::torch_ext::TritonKernelLaunchOp launchOp,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::OperandRange kernelOperands = launchOp.getKernelOperands();
    llvm::SmallVector<mlir::Value> rewrittenKernelOperands;
    rewrittenKernelOperands.reserve(kernelOperands.size());
    bool changed = false;
    rewriter.setInsertionPoint(launchOp);
    for (const auto &indexedOperand : llvm::enumerate(kernelOperands)) {
      mlir::Value originalValue = indexedOperand.value();
      mlir::Value rewrittenValue = rewriteKernelOperand(originalValue);
      if (rewrittenValue != originalValue) {
        changed = true;
      }
      rewrittenKernelOperands.push_back(rewrittenValue);
    }

    if (!changed) {
      return mlir::failure();
    }

    mlir::OperationState state(
        launchOp.getLoc(),
        libtriton::torch_ext::TritonKernelLaunchOp::getOperationName());
    libtriton::torch_ext::TritonKernelLaunchOp::build(
        rewriter, state, launchOp.getKernelAttr(), launchOp.getGridSizeX(),
        launchOp.getGridSizeY(), launchOp.getGridSizeZ(),
        launchOp.getBlockSizeX(), launchOp.getBlockSizeY(),
        launchOp.getBlockSizeZ(), launchOp.getDynamicSharedMemorySize(),
        rewrittenKernelOperands, launchOp.getAsyncObject());
    mlir::Operation *newLaunchOp = rewriter.create(state);
    rewriter.replaceOp(launchOp, newLaunchOp->getResults());
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
