//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "libtriton-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/Type.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"

#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "tvm/ffi/c_api.h"

namespace libtriton::torchext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOGPU
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

/// Converts torch_ext.triton_kernel_launch to gpu.launch_func.
class ConvertTritonKernelLaunchOp
    : public mlir::OpConversionPattern<TritonKernelLaunchOp> {
public:
  ConvertTritonKernelLaunchOp(mlir::TypeConverter &typeConverter,
                              mlir::MLIRContext *context)
      : mlir::OpConversionPattern<TritonKernelLaunchOp>(typeConverter,
                                                        context) {}

  mlir::LogicalResult
  matchAndRewrite(TritonKernelLaunchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::IntegerType i8Ty = mlir::IntegerType::get(ctx, 8);
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType dlTensorTy =
        libtriton::conversion::utils::getDLTensorType(ctx);

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
        mlir::Value dlTensorPtr = mlir::LLVM::GEPOp::create(
            rewriter, loc, ptrTy, i8Ty, adapted,
            mlir::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
        mlir::Value dataGep = mlir::LLVM::GEPOp::create(
            rewriter, loc, ptrTy, dlTensorTy, dlTensorPtr,
            mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
        operands.push_back(
            mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, dataGep));
      } else {
        operands.push_back(adapted);
      }
    }

    // Retrieve the current CUDA stream and pass as asyncObject.
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op->emitOpError("op is not inside a ModuleOp");
    }

    // Step 1: call aoti_torch_get_current_device_index(&slot).
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDevIdxFn =
        libtriton::conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(
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
        libtriton::conversion::utils::getOrCreateAOTITorchGetCurrentStream(
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

    target.addIllegalOp<TritonKernelLaunchOp>();
    target.addLegalDialect<mlir::gpu::GPUDialect, mlir::BuiltinDialect,
                           mlir::LLVM::LLVMDialect>();

    populateTorchExtToGPUConversionPatterns(target, patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct TorchExtToGPUDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateTorchExtToGPUConversionPatterns(target, patterns, typeConverter);
  }
};

} // namespace

void populateTorchExtToGPUConversionPatterns(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter) {
  patterns.add<ConvertTritonKernelLaunchOp>(typeConverter,
                                            patterns.getContext());
}

void registerConvertTorchExtToGPUInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *, libtriton::torchext::TorchExtDialect *dialect) {
        dialect->addInterfaces<TorchExtToGPUDialectInterface>();
      });
}

} // namespace libtriton::torchext
