#include "libtriton-core/Dialect/TritonRT/Transforms/BufferizableOpInterfaceImpl.h"

#include "libtriton-core/Dialect/TritonRT/IR/TritonRTDialect.h"
#include "libtriton-core/Dialect/TritonRT/IR/TritonRTOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace libtriton::triton_rt {

namespace {

/// Bufferization external model for `triton_rt.triton_kernel_launch`.
///
/// The op passes tensor operands by value to a Triton kernel; the kernel reads
/// them but never writes back through those pointers from the caller's
/// perspective. After bufferization every tensor operand is replaced by its
/// corresponding memref buffer, turning the op into a fully memref-based
/// launch.
struct TritonKernelLaunchOpInterface
    : public BufferizableOpInterface::ExternalModel<
          TritonKernelLaunchOpInterface,
          libtriton::triton_rt::TritonKernelLaunchOp> {

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
    auto launchOp = cast<libtriton::triton_rt::TritonKernelLaunchOp>(op);

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
    replaceOpWithNewBufferizedOp<libtriton::triton_rt::TritonKernelLaunchOp>(
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
      +[](MLIRContext *ctx, libtriton::triton_rt::TritonRTDialect *) {
        libtriton::triton_rt::TritonKernelLaunchOp::attachInterface<
            TritonKernelLaunchOpInterface>(*ctx);
      });
}

} // namespace libtriton::triton_rt
