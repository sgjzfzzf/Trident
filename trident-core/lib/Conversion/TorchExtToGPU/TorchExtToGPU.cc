//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "trident-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident-core/Conversion/Utils/Type.h"
#include "trident-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "tvm/ffi/c_api.h"

namespace trident::torchext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOGPU
#include "trident-core/Conversion/Passes.h.inc"

namespace {

/// Converts torch_ext.trident_kernel_launch to gpu.launch_func.
class ConvertTridentKernelLaunchOp
    : public mlir::OpConversionPattern<TridentKernelLaunchOp> {
public:
  ConvertTridentKernelLaunchOp(mlir::TypeConverter &typeConverter,
                               mlir::MLIRContext *context)
      : mlir::OpConversionPattern<TridentKernelLaunchOp>(typeConverter,
                                                         context) {}

  mlir::LogicalResult
  matchAndRewrite(TridentKernelLaunchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::IntegerType i1Ty = mlir::IntegerType::get(ctx, 1);
    mlir::IntegerType i8Ty = mlir::IntegerType::get(ctx, 8);
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType dlTensorTy =
        trident::conversion::utils::getDLTensorType(ctx);

    // Build grid and block dimensions from individual values.
    // Use the original op values (index type) rather than adapted (i64),
    // because gpu.launch_func requires index for grid/block sizes.
    mlir::gpu::KernelDim3 gridSize{op.getGridSizeX(), op.getGridSizeY(),
                                   op.getGridSizeZ()};
    mlir::gpu::KernelDim3 blockSize{op.getBlockSizeX(), op.getBlockSizeY(),
                                    op.getBlockSizeZ()};

    // Build optional cluster dimensions.
    std::optional<mlir::gpu::KernelDim3> clusterSize = std::nullopt;
    if (op.getClusterSizeX() && op.getClusterSizeY() && op.getClusterSizeZ()) {
      clusterSize = mlir::gpu::KernelDim3{
          op.getClusterSizeX(), op.getClusterSizeY(), op.getClusterSizeZ()};
    }

    // Dynamic shared memory size (may be null) — use adapted value for i32.
    mlir::Value dynamicSharedMemorySize = adaptor.getDynamicSharedMemorySize();

    // Type-convert kernel operands.  For tensor operands, the adapted value is
    // a TVMFFIObjectHandle (!llvm.ptr); extract the DLTensor data pointer.
    // Scalar types (i64, f64, etc.) are passed through as-is from the adapter.
    llvm::SmallVector<mlir::Value> operands;
    for (auto [orig, adapted] :
         llvm::zip(op.getKernelOperands(), adaptor.getKernelOperands())) {
      if (mlir::isa<mlir::torch::Torch::BaseTensorType>(orig.getType())) {
        // Extract pointer from TVMFFIAny field[2].
        mlir::Value handleInt =
            mlir::LLVM::ExtractValueOp::create(rewriter, loc, adapted,
                                               llvm::ArrayRef<int64_t>{2})
                .getResult();
        mlir::Value handle =
            mlir::LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, handleInt);
        // Skip 24-byte TVMFFIObject header to reach DLTensor.
        mlir::Value dlTensorPtr = mlir::LLVM::GEPOp::create(
            rewriter, loc, ptrTy, i8Ty, handle,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
        mlir::Value dataGep = mlir::LLVM::GEPOp::create(
            rewriter, loc, ptrTy, dlTensorTy, dlTensorPtr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
        operands.push_back(
            mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, dataGep));
      } else if (mlir::isa<mlir::torch::Torch::BoolType>(orig.getType())) {
        // Bool: extract i64 payload and truncate to i1.
        mlir::Value payload =
            mlir::LLVM::ExtractValueOp::create(rewriter, loc, adapted,
                                               llvm::ArrayRef<int64_t>{2})
                .getResult();
        operands.push_back(
            mlir::LLVM::TruncOp::create(rewriter, loc, i1Ty, payload));
      } else {
        // Scalar: extract i64 payload from TVMFFIAny field[2].
        mlir::Value payload =
            mlir::LLVM::ExtractValueOp::create(rewriter, loc, adapted,
                                               llvm::ArrayRef<int64_t>{2})
                .getResult();
        operands.push_back(payload);
      }
    }

    // Retrieve the current CUDA stream and pass as asyncObject.
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op->emitOpError("op is not inside a ModuleOp");
    }

    // Step 1: call aoti_torch_get_current_device_index(&slot).
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDevIdxFn =
        trident::conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(
            moduleOp);
    if (mlir::failed(getDevIdxFn)) {
      return op->emitOpError(
          "failed to create aoti_torch_get_current_device_index");
    }

    mlir::Value devIdxSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, i32Ty,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
    mlir::LLVM::CallOp::create(rewriter, loc, *getDevIdxFn,
                               mlir::ValueRange{devIdxSlot});
    mlir::Value deviceIndex =
        mlir::LLVM::LoadOp::create(rewriter, loc, i32Ty, devIdxSlot);

    // Step 2: call aoti_torch_get_current_stream(deviceIndex, &slot).
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getStreamFn =
        trident::conversion::utils::getOrCreateAOTITorchGetCurrentStream(
            moduleOp);
    if (mlir::failed(getStreamFn)) {
      return op->emitOpError("failed to create aoti_torch_get_current_stream");
    }

    mlir::Value streamSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
    mlir::LLVM::CallOp::create(rewriter, loc, *getStreamFn,
                               mlir::ValueRange{deviceIndex, streamSlot});
    mlir::Value asyncObject =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, streamSlot);

    // Triton kernels always include 2 extra u64 pointer parameters in the
    // PTX parameter list beyond the user-visible runtime parameters.
    // These are never loaded by the kernel body but cuLaunchKernel still
    // reads them from the params array.  Pad with null (zero) values to
    // match the kernel's actual parameter count and avoid out-of-bounds
    // reads that cause a segfault.
    mlir::Value nullPtr =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
    operands.push_back(nullPtr);
    operands.push_back(nullPtr);

    // Create gpu.launch_func with the current stream as asyncObject.
    rewriter.replaceOpWithNewOp<mlir::gpu::LaunchFuncOp>(
        op, op.getKernel(), gridSize, blockSize, dynamicSharedMemorySize,
        operands, asyncObject, clusterSize);

    return mlir::success();
  }
};

class ConvertTorchExtToGPUPass
    : public impl::ConvertTorchExtToGPUBase<ConvertTorchExtToGPUPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::TypeConverter typeConverter;
    mlir::RewritePatternSet patterns(&getContext());

    // Identity fallback so that index and other unlisted types pass through.
    typeConverter.addConversion([](mlir::Type type) { return type; });
    // Reuse backend type conversion: tensors -> !llvm.ptr, int -> i64, etc.
    // (registered after identity so specific conversions take priority).
    torch::setupBackendTypeConversion(target, typeConverter);

    target.addIllegalOp<TridentKernelLaunchOp>();
    target.addLegalDialect<mlir::gpu::GPUDialect, mlir::BuiltinDialect,
                           mlir::LLVM::LLVMDialect>();

    populateTorchExtToGPUConversionPatterns(target, patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTorchExtToGPUConversionPatterns(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter) {
  patterns.add<ConvertTridentKernelLaunchOp>(typeConverter,
                                             patterns.getContext());
}

} // namespace trident::torchext
