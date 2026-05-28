#include "mlir/IR/MLIRContext.h"
#define GEN_PASS_DEF_ASYNCKERNELLAUNCH
#include "libtriton-core/Dialect/TorchExt/Transforms/TorchExtAsyncKernelLaunch.h"

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::torch_ext {

namespace {

static mlir::Value createCurrentStreamValue(mlir::OpBuilder &builder,
                                            mlir::Location loc) {
  return GetCurrentStreamOp::create(
             builder, loc, mlir::gpu::AsyncTokenType::get(builder.getContext()))
      .getOutput();
}

class GpuLaunchFuncAsyncPattern
    : public mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp> {
public:
  using mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp launchOp,
                  mlir::PatternRewriter &rewriter) const final {
    if (launchOp.getAsyncToken()) {
      return mlir::failure();
    }
    std::optional<mlir::gpu::KernelDim3> clusterSize = std::nullopt;
    if (launchOp.hasClusterSize()) {
      clusterSize = launchOp.getClusterSizeOperandValues();
    }
    rewriter.replaceOpWithNewOp<mlir::gpu::LaunchFuncOp>(
        launchOp, launchOp.getKernelAttr(), launchOp.getGridSizeOperandValues(),
        launchOp.getBlockSizeOperandValues(),
        launchOp.getDynamicSharedMemorySize(), launchOp.getKernelOperands(),
        mlir::gpu::AsyncTokenType::get(rewriter.getContext()),
        llvm::SmallVector<mlir::Value>{
            createCurrentStreamValue(rewriter, launchOp.getLoc())},
        clusterSize);
    return mlir::success();
  }
};

class TritonLaunchAsyncTokenPattern
    : public mlir::OpRewritePattern<TritonKernelLaunchOp> {
public:
  using mlir::OpRewritePattern<TritonKernelLaunchOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TritonKernelLaunchOp launchOp,
                  mlir::PatternRewriter &rewriter) const final {
    if (launchOp.getAsyncToken()) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<TritonKernelLaunchOp>(
        launchOp, mlir::gpu::AsyncTokenType::get(rewriter.getContext()),
        llvm::SmallVector<mlir::Value>{
            createCurrentStreamValue(rewriter, launchOp.getLoc())},
        launchOp.getKernelAttr(), launchOp.getGridSizeX(),
        launchOp.getGridSizeY(), launchOp.getGridSizeZ(),
        launchOp.getBlockSizeX(), launchOp.getBlockSizeY(),
        launchOp.getBlockSizeZ(), launchOp.getClusterSizeX(),
        launchOp.getClusterSizeY(), launchOp.getClusterSizeZ(),
        launchOp.getDynamicSharedMemorySize(), launchOp.getKernelOperands(),
        launchOp.getAsyncObject());
    return mlir::success();
  }
};

class AsyncKernelLaunchPass
    : public impl::AsyncKernelLaunchBase<AsyncKernelLaunchPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ModuleOp moduleOp = getOperation();
    mlir::RewritePatternSet patterns(&context);
    patterns.add<GpuLaunchFuncAsyncPattern, TritonLaunchAsyncTokenPattern>(
        &context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace libtriton::torch_ext
