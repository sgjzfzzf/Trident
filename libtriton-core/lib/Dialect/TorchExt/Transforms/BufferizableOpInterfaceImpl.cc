#include "libtriton-core/Dialect/TorchExt/Transforms/BufferizableOpInterfaceImpl.h"

#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace libtriton::torch_ext {

namespace {

/// Bufferization external model for `torch_ext.torch_kernel_launch`.
///
/// The op passes tensor operands by value to a Torch kernel; the kernel reads
/// them but never writes back through those pointers from the caller's
/// perspective. After bufferization every tensor operand is replaced by its
/// corresponding memref buffer, turning the op into a fully memref-based
/// launch.
struct TorchKernelLaunchOpInterface
    : public BufferizableOpInterface::ExternalModel<
          TorchKernelLaunchOpInterface,
          libtriton::torch_ext::TorchKernelLaunchOp> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // The kernel reads every tensor operand.
    return isa<TensorType>(opOperand.get().getType());
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // TODO: Conservatively treat every tensor operand as a potential write.
    // This matches the read-side heuristic and avoids incorrect copy elision.
    // A more precise implementation should inspect whether the corresponding
    // fx graph operand was produced via a `tensor_to_clone`-style injection
    // (indicating the kernel may mutate the underlying buffer in-place) and
    // only return true in that case.
    return isa<TensorType>(opOperand.get().getType());
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // The op has no results, so no aliasing relationships.
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto launchOp = cast<libtriton::torch_ext::TorchKernelLaunchOp>(op);

    // Replace every tensor-typed kernel operand with its memref buffer.
    llvm::SmallVector<Value> newKernelOperands;
    newKernelOperands.reserve(launchOp.getKernelOperands().size());
    for (Value operand : launchOp.getKernelOperands()) {
      if (isa<TensorType>(operand.getType())) {
        FailureOr<Value> buffer = getBuffer(rewriter, operand, options, state);
        if (failed(buffer))
          return failure();
        newKernelOperands.push_back(*buffer);
      } else {
        newKernelOperands.push_back(operand);
      }
    }

    // Recreate the op with the updated operands. The op has no results, so
    // replaceOpWithNewBufferizedOp correctly calls
    // replaceOpWithBufferizedValues with an empty result list and erases the
    // original op.
    replaceOpWithNewBufferizedOp<libtriton::torch_ext::TorchKernelLaunchOp>(
        rewriter, op, launchOp.getKernelAttr(), launchOp.getGridSizeX(),
        launchOp.getGridSizeY(), launchOp.getGridSizeZ(),
        launchOp.getBlockSizeX(), launchOp.getBlockSizeY(),
        launchOp.getBlockSizeZ(), launchOp.getDynamicSharedMemorySize(),
        newKernelOperands);
    return success();
  }
};

} // namespace

void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, libtriton::torch_ext::TorchExtDialect *) {
        libtriton::torch_ext::TorchKernelLaunchOp::attachInterface<
            TorchKernelLaunchOpInterface>(*ctx);
      });
}

} // namespace libtriton::torch_ext
