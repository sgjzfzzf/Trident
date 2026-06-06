#include "libtriton-core/Conversion/TorchToArith/TorchToArith.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOARITH
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

/// Converts torch.constant.bool to arith.constant.
class ConvertTorchConstantBoolOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantBoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantBoolOp::Adaptor;
  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantBoolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, op.getValueAttr());
    return mlir::success();
  }
};

/// Converts torch.constant.int to arith.constant.
/// arith.constant only accepts signless integers, so the signed !torch.int
/// value is converted to a signless i64 attribute.
class ConvertTorchConstantIntOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantIntOp::Adaptor;
  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, rewriter.getIntegerAttr(rewriter.getI64Type(),
                                    op.getValueAttr().getValue()));
    return mlir::success();
  }
};

/// Converts torch.constant.float to arith.constant.
class ConvertTorchConstantFloatOp
    : public mlir::OpConversionPattern<mlir::torch::Torch::ConstantFloatOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = mlir::torch::Torch::ConstantFloatOp::Adaptor;
  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ConstantFloatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, op.getValueAttr());
    return mlir::success();
  }
};

class ConvertTorchToArithPass
    : public impl::ConvertTorchToArithBase<ConvertTorchToArithPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    populateTorchToArithConversionPatterns(target, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTorchToArithConversionPatterns(mlir::ConversionTarget &target,
                                            mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertTorchConstantBoolOp, ConvertTorchConstantIntOp,
               ConvertTorchConstantFloatOp>(patterns.getContext());

  target.addIllegalOp<mlir::torch::Torch::ConstantBoolOp,
                      mlir::torch::Torch::ConstantIntOp,
                      mlir::torch::Torch::ConstantFloatOp>();
  target.addLegalDialect<mlir::arith::ArithDialect, mlir::BuiltinDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

} // namespace libtriton::torch
