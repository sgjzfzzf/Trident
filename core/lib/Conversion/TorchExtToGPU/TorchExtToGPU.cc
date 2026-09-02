//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchExtToGPU/TorchExtToGPU.h"
#include "dlpack/dlpack.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <utility>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOGPU
#include "trident/core/Conversion/Passes.h.inc"

/// Converts torch_ext.trident_kernel_launch to gpu.launch_func.
class ConvertTridentKernelLaunchOp
    : public mlir::OpConversionPattern<torchext::TridentKernelLaunchOp> {
public:
  ConvertTridentKernelLaunchOp(mlir::TypeConverter &typeConverter,
                               mlir::MLIRContext *context)
      : mlir::OpConversionPattern<torchext::TridentKernelLaunchOp>(
            typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(torchext::TridentKernelLaunchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::IntegerType const i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType const i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(ctx);

    // Build grid and block dimensions from individual i64 values.
    // When all operand types are already legal LLVM types, the
    // GpuToLLVMConversionPass marks the op dynamically legal and
    // LegalizeLaunchFuncOpPattern never matches — this preserves the
    // asyncObject (current CUDA stream) through the lowering pipeline.
    mlir::gpu::KernelDim3 const gridSize{
        adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
    mlir::gpu::KernelDim3 const blockSize{adaptor.getBlockSizeX(),
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
    mlir::Value const dynamicSharedMemorySize =
        adaptor.getDynamicSharedMemorySize();

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op->emitOpError("op is not inside a ModuleOp");
    }

    // Triton kernels are compiled into a gpu.binary.  The kernel name after
    // the root reference is stored in the binary and is resolved later by
    // gpu.launch_func lowering/runtime.
    mlir::gpu::BinaryOp binary = moduleOp.lookupSymbol<mlir::gpu::BinaryOp>(
        op.getKernel().getRootReference());
    if (!binary) {
      return op->emitOpError("cannot find GPU binary ")
             << op.getKernel().getRootReference();
    }

    llvm::SmallVector<mlir::Value> operands(adaptor.getKernelOperands().begin(),
                                            adaptor.getKernelOperands().end());

    // Retrieve the current CUDA stream and pass as asyncObject.
    // Step 1: call aoti_torch_get_current_device_index(&slot).
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getDevIdxFn =
        utils::getOrCreateAOTITorchGetCurrentDeviceIndex(moduleOp);
    if (mlir::failed(getDevIdxFn)) {
      return op->emitOpError(
          "failed to create aoti_torch_get_current_device_index");
    }

    mlir::Value const devIdxSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, i32Ty,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
    mlir::LLVM::CallOp::create(rewriter, loc, getDevIdxFn.value(),
                               mlir::ValueRange{devIdxSlot});
    mlir::Value const deviceIndex =
        mlir::LLVM::LoadOp::create(rewriter, loc, i32Ty, devIdxSlot);

    // Step 2: call TVMFFIEnvGetStream(kDLCUDA, deviceIndex) to get the
    // current CUDA stream handle directly (returns void*).
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getStreamFn =
        utils::getOrCreateTVMFFIEnvGetStream(moduleOp);
    if (mlir::failed(getStreamFn)) {
      return op->emitOpError("failed to create TVMFFIEnvGetStream");
    }

    mlir::Value const cudaDeviceType = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, DLDeviceType::kDLCUDA);
    mlir::Value const asyncObject =
        mlir::LLVM::CallOp::create(
            rewriter, loc, getStreamFn.value(),
            mlir::ValueRange{cudaDeviceType, deviceIndex})
            .getResult();

    // Triton kernels always include 2 extra u64 pointer parameters in the
    // PTX parameter list beyond the user-visible runtime parameters.
    // These are never loaded by the kernel body but cuLaunchKernel still
    // reads them from the params array.  Pad with null (zero) values to
    // match the kernel's actual parameter count and avoid out-of-bounds
    // reads that cause a segfault.
    mlir::Value const nullPtr =
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
    typeConverter.addConversion(
        [](mlir::torch::Torch::BoolType type) -> mlir::Type {
          return mlir::IntegerType::get(type.getContext(), 1);
        });
    typeConverter.addConversion(
        [](mlir::torch::Torch::FloatType type) -> mlir::Type {
          return mlir::Float32Type::get(type.getContext());
        });
    typeConverter.addConversion(
        [](mlir::torch::Torch::IntType type) -> mlir::Type {
          return mlir::IntegerType::get(type.getContext(), 32);
        });
    typeConverter.addConversion(
        [](mlir::torch::Torch::NonValueTensorType type) -> mlir::Type {
          return mlir::LLVM::LLVMPointerType::get(type.getContext());
        });
    typeConverter.addConversion(
        [](mlir::torch::Torch::ValueTensorType type) -> mlir::Type {
          return mlir::LLVM::LLVMPointerType::get(type.getContext());
        });
    typeConverter.addTargetMaterialization(
        [](mlir::OpBuilder &builder, mlir::Type resultType,
           mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
          mlir::Value const input = inputs.front();
          mlir::Type const inputType = input.getType();
          if (mlir::isa<mlir::torch::Torch::BaseTensorType>(inputType) &&
              mlir::isa<mlir::LLVM::LLVMPointerType>(resultType)) {
            mlir::LLVM::LLVMPointerType const pointerType =
                mlir::cast<mlir::LLVM::LLVMPointerType>(resultType);
            mlir::Value const object = torchext::GetOp::create(
                builder, loc,
                tvm_ffi::ObjectType::get(pointerType.getContext()), input);
            mlir::Value const tensor = tvm_ffi::AsOp::create(
                builder, loc,
                dlpack::DLTensorType::get(pointerType.getContext()), object);
            return dlpack::TensorDataOp::create(builder, loc, pointerType,
                                                tensor)
                .getResult();
          } else if (mlir::isa<mlir::torch::Torch::BoolType>(inputType) &&
                     resultType.isSignlessInteger()) {
            return torchext::GetOp::create(builder, loc, resultType, input)
                .getResult();
          } else if (mlir::isa<mlir::torch::Torch::IntType>(inputType) &&
                     resultType.isSignlessInteger()) {
            return mlir::LLVM::TruncOp::create(
                       builder, loc, resultType,
                       torchext::GetOp::create(
                           builder, loc,
                           mlir::IntegerType::get(inputType.getContext(), 64),
                           input))
                .getResult();
          } else if (mlir::isa<mlir::torch::Torch::FloatType>(inputType) &&
                     mlir::isa<mlir::FloatType>(resultType)) {
            return mlir::LLVM::FPTruncOp::create(
                       builder, loc, resultType,
                       torchext::GetOp::create(
                           builder, loc,
                           mlir::Float64Type::get(inputType.getContext()),
                           input))
                .getResult();
          } else {
            return mlir::UnrealizedConversionCastOp::create(
                       builder, loc, mlir::TypeRange(resultType), input)
                .getResult(0);
          }
        });

    typeConverter.addConversion(
        [](mlir::Type type) -> std::optional<mlir::Type> {
          if (mlir::isa<mlir::IntegerType, mlir::FloatType>(type)) {
            return type;
          } else {
            return std::nullopt;
          }
        });
    target.addIllegalOp<torchext::TridentKernelLaunchOp>();
    target.addLegalOp<mlir::gpu::LaunchFuncOp, torchext::GetOp>();
    target.addLegalDialect<mlir::gpu::GPUDialect, mlir::BuiltinDialect,
                           mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                           tvm_ffi::TVMFFIDialect, dlpack::DLPackDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    populateTorchExtToGPUConversionPatterns(target, patterns, typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

void populateTorchExtToGPUConversionPatterns(
    mlir::ConversionTarget &, mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter) {
  patterns.add<ConvertTridentKernelLaunchOp>(typeConverter,
                                             patterns.getContext());
}

} // namespace trident::conversion
