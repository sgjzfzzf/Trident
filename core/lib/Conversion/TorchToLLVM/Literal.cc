//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "trident/core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/TridentCAPIDescriptors.h"
#include "tvm/ffi/c_api.h"

namespace trident::torch {
namespace {

/// Lowers torch.vtensor.literal dense single-value constants to
/// aoti_torch_aten_full(size, fill, dtype, ...).
class ConvertTorchValueTensorLiteralOp
    : public mlir::OpConversionPattern<
          mlir::torch::Torch::ValueTensorLiteralOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ValueTensorLiteralOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ValueTensorLiteralOp op,
                  [[maybe_unused]] OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    mlir::DenseElementsAttr dense =
        mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    if (!dense) {
      return op.emitError("torch.vtensor.literal must use DenseElementsAttr");
    }

    mlir::torch::Torch::BaseTensorType tensorTy =
        mlir::dyn_cast<mlir::torch::Torch::BaseTensorType>(op.getType());
    if (!tensorTy || !tensorTy.hasSizes()) {
      return op.emitError(
          "torch.vtensor.literal requires statically known result shape");
    }

    mlir::Type elementType = dense.getElementType();
    llvm::ArrayRef<int64_t> shape = tensorTy.getSizes();
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::IntegerType i8Ty = mlir::IntegerType::get(ctx, 8);
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());

    mlir::LLVM::LLVMFuncOp toTorchDtypeFn = TRIDENT_CHECK_FAILURE(
        trident::conversion::utils::getOrCreateTVMFFIToTorchType(moduleOp));
    mlir::LLVM::LLVMFuncOp toTorchDeviceTypeFn = TRIDENT_CHECK_FAILURE(
        trident::conversion::utils::getOrCreateTVMFFIDeviceToTorchDeviceType(
            moduleOp));

    mlir::LLVM::LLVMFuncOp packTensorFn = TRIDENT_CHECK_FAILURE(
        trident::conversion::utils::getOrCreateTensorToTVMFFIObject(moduleOp));

    uint8_t dlCode = 0;
    uint8_t dlBits = 0;
    if (mlir::isa<mlir::Float16Type>(elementType)) {
      dlCode = kDLFloat;
      dlBits = 16;
    } else if (mlir::isa<mlir::BFloat16Type>(elementType)) {
      dlCode = kDLBfloat;
      dlBits = 16;
    } else if (mlir::isa<mlir::Float32Type>(elementType)) {
      dlCode = kDLFloat;
      dlBits = 32;
    } else if (mlir::isa<mlir::Float64Type>(elementType)) {
      dlCode = kDLFloat;
      dlBits = 64;
    } else if (mlir::IntegerType intTy =
                   mlir::dyn_cast<mlir::IntegerType>(elementType)) {
      if (intTy.isSignlessInteger(1)) {
        dlCode = kDLBool;
        dlBits = 8;
      } else if (intTy.isUnsignedInteger()) {
        dlCode = kDLUInt;
        dlBits = intTy.getWidth();
      } else {
        dlCode = kDLInt;
        dlBits = intTy.getWidth();
      }
    }
    if (dlBits == 0 || dlCode == 0) {
      return op.emitError("unsupported torch.vtensor.literal dtype");
    }

    mlir::Value dtypeVal =
        mlir::LLVM::CallOp::create(
            rewriter, loc, toTorchDtypeFn,
            {mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty, dlCode),
             mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty, dlBits)})
            .getResult();

    mlir::Value rankVal =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, shape.size());
    mlir::Value sizePtr = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, shape.size()));
    for (auto [idx, dim] : llvm::enumerate(shape)) {
      mlir::Value slot =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty, sizePtr,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{idx});
      mlir::LLVM::StoreOp::create(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, dim), slot);
    }

    mlir::Value tensorHandle;

    if (dense.isSplat()) {
      double fillValue;
      if (mlir::isa<mlir::FloatType>(elementType)) {
        llvm::APFloat ap = dense.getSplatValue<llvm::APFloat>();
        fillValue = ap.convertToDouble();
      } else if (mlir::IntegerType intTy =
                     mlir::dyn_cast<mlir::IntegerType>(elementType)) {
        llvm::APInt ap = dense.getSplatValue<llvm::APInt>();
        if (intTy.isUnsignedInteger()) {
          fillValue = ap.getZExtValue();
        } else if (intTy.isSignlessInteger(1)) {
          fillValue = ap.getBoolValue();
        } else {
          fillValue = ap.getSExtValue();
        }
      } else {
        return op.emitError("unsupported torch.vtensor.literal element type");
      }

      mlir::LLVM::LLVMFuncOp fullFn = TRIDENT_CHECK_FAILURE(
          trident::conversion::utils::getOrCreateAOTITorchAtenFull(moduleOp));
      mlir::LLVM::LLVMFuncOp getDeviceIndexFn = TRIDENT_CHECK_FAILURE(
          trident::conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(
              moduleOp));

      mlir::Value oneI64 =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
      mlir::Value zeroI32 =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);
      mlir::Value cudaDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kDLCUDA);
      mlir::Value cudaDeviceType =
          mlir::LLVM::CallOp::create(rewriter, loc, toTorchDeviceTypeFn,
                                     mlir::ValueRange{cudaDLDeviceType})
              .getResult();

      mlir::Value dtypeSlot =
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i32Ty, oneI64);
      mlir::LLVM::StoreOp::create(rewriter, loc, dtypeVal, dtypeSlot);

      mlir::Value deviceTypeSlot =
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i32Ty, oneI64);
      mlir::LLVM::StoreOp::create(rewriter, loc, cudaDeviceType,
                                  deviceTypeSlot);

      mlir::Value currentDeviceIndexSlot =
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i32Ty, oneI64);
      mlir::LLVM::CallOp::create(rewriter, loc, getDeviceIndexFn,
                                 mlir::ValueRange{currentDeviceIndexSlot});
      mlir::Value currentDeviceIndex = mlir::LLVM::LoadOp::create(
          rewriter, loc, i32Ty, currentDeviceIndexSlot);

      mlir::Value outTensorSlot =
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, oneI64);
      mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy);
      mlir::Value fillVal = mlir::LLVM::ConstantOp::create(
          rewriter, loc, mlir::Float64Type::get(ctx),
          mlir::FloatAttr::get(mlir::Float64Type::get(ctx), fillValue));

      mlir::LLVM::CallOp::create(
          rewriter, loc, fullFn,
          mlir::ValueRange{sizePtr, rankVal, fillVal, dtypeSlot, nullPtr,
                           deviceTypeSlot, currentDeviceIndex, nullPtr,
                           outTensorSlot});

      tensorHandle =
          mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, outTensorSlot);
    } else {
      mlir::RankedTensorType denseType =
          mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
      if (!denseType || denseType.getShape() != shape) {
        return op.emitError(
            "torch.vtensor.literal dense shape must match result type shape");
      }

      int64_t storageBits = -1;
      if (mlir::FloatType floatTy =
              mlir::dyn_cast<mlir::FloatType>(elementType)) {
        storageBits = floatTy.getWidth();
      } else if (mlir::IntegerType intTy =
                     mlir::dyn_cast<mlir::IntegerType>(elementType)) {
        storageBits = intTy.isSignlessInteger(1) ? 8 : intTy.getWidth();
      }
      if (storageBits <= 0) {
        return op.emitError("unsupported torch.vtensor.literal element type");
      }

      mlir::LLVM::LLVMFuncOp createFromBlobFn = TRIDENT_CHECK_FAILURE(
          trident::conversion::utils::getOrCreateAOTITorchCreateTensorFromBlob(
              moduleOp));
      mlir::LLVM::LLVMFuncOp emptyStridedFn = TRIDENT_CHECK_FAILURE(
          trident::conversion::utils::getOrCreateAOTITorchEmptyStrided(
              moduleOp));
      mlir::LLVM::LLVMFuncOp copyFn = TRIDENT_CHECK_FAILURE(
          trident::conversion::utils::getOrCreateAOTITorchCopy_(moduleOp));
      mlir::LLVM::LLVMFuncOp deleteTensorFn = TRIDENT_CHECK_FAILURE(
          trident::conversion::utils::getOrCreateAOTITorchDeleteTensorObject(
              moduleOp));
      mlir::LLVM::LLVMFuncOp getDeviceIndexFn = TRIDENT_CHECK_FAILURE(
          trident::conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(
              moduleOp));

      mlir::IntegerType storageTy = mlir::IntegerType::get(ctx, storageBits);
      mlir::Value stridePtr = mlir::LLVM::AllocaOp::create(
          rewriter, loc, ptrTy, i64Ty,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, shape.size()));
      for (int64_t stride = 1, idx = shape.size() - 1; idx >= 0;
           stride *= shape[idx--]) {
        mlir::Value slot =
            mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, i64Ty, stridePtr,
                                      llvm::ArrayRef<mlir::LLVM::GEPArg>{idx});
        mlir::LLVM::StoreOp::create(
            rewriter, loc,
            mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, stride), slot);
      }

      mlir::Value blobDataPtr = mlir::LLVM::AllocaOp::create(
          rewriter, loc, ptrTy, storageTy,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                         dense.getNumElements()));

      auto storeElementAt = [&](size_t index, llvm::APInt bitsValue) {
        mlir::Value slot = mlir::LLVM::GEPOp::create(
            rewriter, loc, ptrTy, storageTy, blobDataPtr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
        mlir::LLVM::StoreOp::create(
            rewriter, loc,
            mlir::LLVM::ConstantOp::create(rewriter, loc, storageTy, bitsValue),
            slot);
      };

      if (mlir::isa<mlir::FloatType>(elementType)) {
        for (auto [idx, valueIt] :
             llvm::enumerate(dense.getValues<llvm::APFloat>())) {
          storeElementAt(idx, valueIt.bitcastToAPInt());
        }
      } else if (mlir::IntegerType intTy =
                     mlir::dyn_cast<mlir::IntegerType>(elementType)) {
        for (auto [idx, valueIt] :
             llvm::enumerate(dense.getValues<llvm::APInt>())) {
          storeElementAt(idx, intTy.isSignlessInteger(1)
                                  ? llvm::APInt(8, valueIt.getBoolValue())
                                  : valueIt);
        }
      } else {
        return op.emitError("unsupported torch.vtensor.literal element type");
      }

      mlir::Value zeroI64 =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
      mlir::Value zeroI32 =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);
      mlir::Value oneI64 =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
      mlir::Value cpuDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kDLCPU);
      // TODO: Target device is hardcoded to CUDA for now. Replace with real
      // device propagation once device semantics are available in this pass.
      mlir::Value cudaDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kDLCUDA);
      mlir::Value cpuDeviceType =
          mlir::LLVM::CallOp::create(rewriter, loc, toTorchDeviceTypeFn,
                                     mlir::ValueRange{cpuDLDeviceType})
              .getResult();
      mlir::Value cudaDeviceType =
          mlir::LLVM::CallOp::create(rewriter, loc, toTorchDeviceTypeFn,
                                     mlir::ValueRange{cudaDLDeviceType})
              .getResult();

      mlir::Value currentDeviceIndexSlot =
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i32Ty, oneI64);
      mlir::LLVM::CallOp::create(rewriter, loc, getDeviceIndexFn,
                                 mlir::ValueRange{currentDeviceIndexSlot});
      mlir::Value currentDeviceIndex = mlir::LLVM::LoadOp::create(
          rewriter, loc, i32Ty, currentDeviceIndexSlot);

      mlir::Value cpuTensorSlot =
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, oneI64);
      mlir::LLVM::CallOp::create(
          rewriter, loc, createFromBlobFn,
          mlir::ValueRange{blobDataPtr, rankVal, sizePtr, stridePtr, zeroI64,
                           dtypeVal, cpuDeviceType, zeroI32, cpuTensorSlot});

      mlir::Value cpuTensorHandle =
          mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, cpuTensorSlot);
      mlir::Value cudaTensorSlot =
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, oneI64);
      mlir::LLVM::CallOp::create(
          rewriter, loc, emptyStridedFn,
          mlir::ValueRange{rankVal, sizePtr, stridePtr, dtypeVal,
                           cudaDeviceType, currentDeviceIndex, cudaTensorSlot});

      mlir::Value cudaTensorHandle =
          mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, cudaTensorSlot);
      mlir::LLVM::CallOp::create(
          rewriter, loc, copyFn,
          mlir::ValueRange{cudaTensorHandle, cpuTensorHandle, zeroI32});
      mlir::LLVM::CallOp::create(rewriter, loc, deleteTensorFn,
                                 mlir::ValueRange{cpuTensorHandle});

      tensorHandle = cudaTensorHandle;
    }

    mlir::Value outObjectSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
    mlir::LLVM::CallOp::create(rewriter, loc, packTensorFn,
                               mlir::ValueRange{tensorHandle, outObjectSlot});

    mlir::Value objectHandle =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, outObjectSlot);
    mlir::Value payload =
        mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, objectHandle);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFITensor);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

} // namespace

void populateTorchToLLVMLiteralConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertTorchValueTensorLiteralOp>(typeConverter,
                                                 patterns.getContext());
  target.addIllegalOp<mlir::torch::Torch::ValueTensorLiteralOp>();
}

} // namespace trident::torch
