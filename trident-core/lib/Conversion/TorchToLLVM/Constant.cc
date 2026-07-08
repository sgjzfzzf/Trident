//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "CAPI.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "trident-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "tvm/ffi/c_api.h"

namespace trident::torch {
namespace {

/// Lowers torch.constant.bool to LLVM TVMFFIAny {kTVMFFIBool, 0, value}.
class ConvertTorchConstantBoolOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantBoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantBoolOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantBoolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIBool);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    mlir::Value payload =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, op.getValue());
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Lowers torch.constant.int to LLVM TVMFFIAny {kTVMFFIInt, 0, value}.
class ConvertTorchConstantIntOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantIntOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIInt);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    mlir::Value payload =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, op.getValue());
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Lowers torch.constant.float to LLVM TVMFFIAny {kTVMFFIFloat, 0, value}.
class ConvertTorchConstantFloatOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantFloatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantFloatOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantFloatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIFloat);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    mlir::Value floatVal = mlir::LLVM::ConstantOp::create(
        rewriter, loc, mlir::Float64Type::get(ctx), op.getValue());
    mlir::Value payload =
        mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty, floatVal);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Lowers torch.constant.none to LLVM TVMFFIAny {kTVMFFINone, 0, 0}.
class ConvertTorchConstantNoneOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantNoneOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantNoneOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantNoneOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFINone);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
    mlir::Value payload =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, payload, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

/// Lowers torch.constant.device to LLVM TVMFFIAny {kTVMFFIDevice, 0, combined}.
/// The device value is encoded as (device_index << 32) | device_type in the
/// i64 payload, matching the TVM FFI kTVMFFIDevice convention.
class ConvertTorchConstantDeviceOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantDeviceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantDeviceOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantDeviceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type anyTy = getTypeConverter()->convertType(op.getType());
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

    // Parse the device string (e.g. "cuda:0", "cpu")
    const DLDevice dlDevice = torchDeviceToDLDevice(op.getValue().data());

    // Build TVMFFIAny: {kTVMFFIDevice=6, 0, (device_index<<32)|device_type}.
    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    mlir::Value typeIdx =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kTVMFFIDevice);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});

    // Encode: (device_index << 32) | device_type.
    mlir::Value devType64 = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i64Ty, dlDevice.device_type);
    mlir::Value devIdx64 = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                                          dlDevice.device_id);
    mlir::Value shifted = mlir::LLVM::ShlOp::create(
        rewriter, loc, devIdx64,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 32));
    mlir::Value combined =
        mlir::LLVM::OrOp::create(rewriter, loc, devType64, shifted);

    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        op, anyTy, result, combined, llvm::ArrayRef<int64_t>{2});
    return mlir::success();
  }
};

} // namespace

void populateTorchToLLVMConstantConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertTorchConstantBoolOp, ConvertTorchConstantIntOp,
               ConvertTorchConstantFloatOp, ConvertTorchConstantNoneOp,
               ConvertTorchConstantDeviceOp>(typeConverter,
                                             patterns.getContext());
  target.addIllegalOp<
      mlir::torch::Torch::ConstantBoolOp, mlir::torch::Torch::ConstantIntOp,
      mlir::torch::Torch::ConstantFloatOp, mlir::torch::Torch::ConstantNoneOp,
      mlir::torch::Torch::ConstantDeviceOp>();
}

} // namespace trident::torch
