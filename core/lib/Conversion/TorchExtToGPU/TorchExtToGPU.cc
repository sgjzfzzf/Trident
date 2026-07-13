//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "tvm/ffi/c_api.h"

namespace trident::torchext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOGPU
#include "trident/core/Conversion/Passes.h.inc"

namespace {

/// Converts torchext.cast to the appropriate LLVM truncation/extension.
class ConvertCastOp : public mlir::OpConversionPattern<CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type resultType = op.getResult().getType();

    // The type converter maps !torch.float / !torch.int to
    // !llvm.struct<(i32, i32, i64)> (TVMFFIAny).  Extract the i64 payload
    // from field[2], which holds the bitcast of the underlying value.
    mlir::Value payload = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getOperand(), llvm::ArrayRef<int64_t>{2});

    if (mlir::isa<mlir::FloatType>(resultType)) {
      // payload is f64 bitcast to i64 — bitcast back to f64 first.
      mlir::Value f64Val = mlir::LLVM::BitcastOp::create(
          rewriter, loc, mlir::Float64Type::get(ctx), payload);
      if (resultType.isF64()) {
        // f64: no conversion needed.
        rewriter.replaceOp(op, f64Val);
      } else {
        // Everything else (f32, f16, bf16, …): truncate from f64.
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPTruncOp>(op, resultType,
                                                           f64Val);
      }
    } else {
      // payload is i64 — truncate to signless, then cast to
      // target signedness if needed (llvm.trunc accepts only signless types).
      uint32_t targetWidth = resultType.getIntOrFloatBitWidth();
      mlir::Type signlessType = mlir::IntegerType::get(ctx, targetWidth);
      mlir::Value converted =
          targetWidth < 64 ? mlir::LLVM::TruncOp::create(rewriter, loc,
                                                         signlessType, payload)
                           : payload;
      if (resultType != signlessType) {
        rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
            op, resultType, converted);
      } else {
        rewriter.replaceOp(op, converted);
      }
    }
    return mlir::success();
  }
};

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
    // Since TridentKernelLaunchOp uses I64 for grid/block/cluster (not Index),
    // all operands are already legal LLVM types.  When all operand types are
    // legal LLVM types the GpuToLLVMConversionPass marks the op dynamically
    // legal and LegalizeLaunchFuncOpPattern never matches — this preserves
    // the asyncObject (current CUDA stream) through the lowering pipeline.
    mlir::gpu::KernelDim3 gridSize{
        adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
    mlir::gpu::KernelDim3 blockSize{adaptor.getBlockSizeX(),
                                    adaptor.getBlockSizeY(),
                                    adaptor.getBlockSizeZ()};

    // Build optional cluster dimensions.
    std::optional<mlir::gpu::KernelDim3> clusterSize = std::nullopt;
    if (adaptor.getClusterSizeX() && adaptor.getClusterSizeY() &&
        adaptor.getClusterSizeZ()) {
      clusterSize = mlir::gpu::KernelDim3{adaptor.getClusterSizeX(),
                                          adaptor.getClusterSizeY(),
                                          adaptor.getClusterSizeZ()};
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
        mlir::Value handleInt = mlir::LLVM::ExtractValueOp::create(
            rewriter, loc, adapted, llvm::ArrayRef<int64_t>{2});
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
        mlir::Value payload = mlir::LLVM::ExtractValueOp::create(
            rewriter, loc, adapted, llvm::ArrayRef<int64_t>{2});
        operands.push_back(
            mlir::LLVM::TruncOp::create(rewriter, loc, i1Ty, payload));
      } else if (mlir::isa<mlir::torch::Torch::FloatType,
                           mlir::torch::Torch::IntType>(orig.getType())) {
        // Torch scalar: extract i64 payload from TVMFFIAny field[2].
        mlir::Value payload = mlir::LLVM::ExtractValueOp::create(
            rewriter, loc, adapted, llvm::ArrayRef<int64_t>{2});
        operands.push_back(payload);
      } else {
        // Native scalar type (f32, i32, etc. from torchext.cast):
        // pass through directly.
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

    // Step 2: call TVMFFIEnvGetStream(kDLCUDA, deviceIndex) to get the
    // current CUDA stream handle directly (returns void*).
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getStreamFn =
        trident::conversion::utils::getOrCreateTVMFFIEnvGetStream(moduleOp);
    if (mlir::failed(getStreamFn)) {
      return op->emitOpError("failed to create TVMFFIEnvGetStream");
    }

    mlir::Value cudaDeviceType = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, DLDeviceType::kDLCUDA);
    mlir::Value asyncObject = mlir::LLVM::CallOp::create(
                                  rewriter, loc, *getStreamFn,
                                  mlir::ValueRange{cudaDeviceType, deviceIndex})
                                  .getResult();

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

    target.addIllegalOp<CastOp, TridentKernelLaunchOp>();
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
  patterns.add<ConvertCastOp, ConvertTridentKernelLaunchOp>(
      typeConverter, patterns.getContext());
}

} // namespace trident::torchext
