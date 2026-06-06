#include "libtriton-core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Lower torch.constant.* ops directly to LLVM::ConstantOp
// ---------------------------------------------------------------------------

/// Lowers torch.constant.bool to LLVM::ConstantOp (i1).
class ConvertTorchConstantBoolOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantBoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantBoolOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantBoolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, ty, rewriter.getIntegerAttr(ty, op.getValue() ? 1 : 0));
    return mlir::success();
  }
};

/// Lowers torch.constant.int to LLVM::ConstantOp (i64).
class ConvertTorchConstantIntOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantIntOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, ty, rewriter.getIntegerAttr(ty, op.getValue()));
    return mlir::success();
  }
};

/// Lowers torch.constant.float to LLVM::ConstantOp (f64).
class ConvertTorchConstantFloatOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantFloatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantFloatOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantFloatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        op, ty, rewriter.getFloatAttr(ty, op.getValue()));
    return mlir::success();
  }
};

// ---------------------------------------------------------------------------
// ConvertTorchToLLVM pass
// ---------------------------------------------------------------------------

class ConvertTorchToLLVMPass
    : public impl::ConvertTorchToLLVMBase<ConvertTorchToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    setupBackendTypeConversion(target, typeConverter);

    target.addIllegalOp<mlir::torch::Torch::ConstantBoolOp,
                        mlir::torch::Torch::ConstantIntOp,
                        mlir::torch::Torch::ConstantFloatOp>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::BuiltinDialect>();

    patterns.add<ConvertTorchConstantBoolOp, ConvertTorchConstantIntOp,
                 ConvertTorchConstantFloatOp>(typeConverter,
                                              patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  setupBackendTypeConversion(target, typeConverter);
  patterns.add<ConvertTorchConstantBoolOp, ConvertTorchConstantIntOp,
               ConvertTorchConstantFloatOp>(typeConverter,
                                            patterns.getContext());
}

} // namespace libtriton::torch
