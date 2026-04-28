#define GEN_PASS_DEF_ASYNCKERNELLAUNCH
#include "libtriton-core/Dialect/TorchExt/Transforms/TorchExtAsyncKernelLaunch.h"

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace libtriton::torch_ext {

namespace {

static mlir::Value createCurrentStreamValue(mlir::Operation *anchor,
                                            mlir::OpBuilder &builder) {
  mlir::MLIRContext *ctx = anchor->getContext();
  mlir::Type streamTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Value noDevice;
  GetCurrentStreamOp getCurrentStreamOp =
      GetCurrentStreamOp::create(builder, anchor->getLoc(), streamTy,
                                 /*device=*/noDevice);
  return getCurrentStreamOp.getOutput();
}

class AsyncKernelLaunchPass
    : public impl::AsyncKernelLaunchBase<AsyncKernelLaunchPass> {
public:
  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](mlir::gpu::LaunchFuncOp launchOp) {
      if (!launchOp.getAsyncObject()) {
        mlir::OpBuilder builder(launchOp);
        mlir::Value stream = createCurrentStreamValue(launchOp, builder);
        launchOp.getAsyncObjectMutable().assign(stream);
      }
    });

    moduleOp.walk([&](TritonKernelLaunchOp launchOp) {
      if (!launchOp.getAsyncObject()) {
        mlir::OpBuilder builder(launchOp);
        mlir::Value stream = createCurrentStreamValue(launchOp, builder);
        launchOp.getAsyncObjectMutable().assign(stream);
      }
    });
  }
};

} // namespace

} // namespace libtriton::torch_ext
