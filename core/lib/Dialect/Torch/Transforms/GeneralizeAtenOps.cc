//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/Torch/Transforms/GeneralizeAtenOps.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h>
#include <utility>

namespace trident::torch {

#define GEN_PASS_DEF_GENERALIZEATENOPS
#include "trident/core/Dialect/Torch/Transforms/Passes.h.inc"

class ConvertAtenAddInt final
    : public mlir::OpConversionPattern<mlir::torch::Torch::AtenAddIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::AtenAddIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value nativeResult = mlir::arith::AddIOp::create(
        rewriter, op.getLoc(),
        trident::torchext::GetOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64Type(), adaptor.getA())
            .getResult(),
        trident::torchext::GetOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64Type(), adaptor.getB())
            .getResult());
    rewriter.replaceOpWithNewOp<mlir::torch::TorchConversion::FromI64Op>(
        op, nativeResult);
    return mlir::success();
  }
};

class ConvertAtenFloordivInt final
    : public mlir::OpConversionPattern<mlir::torch::Torch::AtenFloordivIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::AtenFloordivIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value nativeResult = mlir::arith::FloorDivSIOp::create(
        rewriter, op.getLoc(),
        trident::torchext::GetOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64Type(), adaptor.getA())
            .getResult(),
        trident::torchext::GetOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64Type(), adaptor.getB())
            .getResult());
    rewriter.replaceOpWithNewOp<mlir::torch::TorchConversion::FromI64Op>(
        op, nativeResult);
    return mlir::success();
  }
};

class ConvertAtenIntBool final
    : public mlir::OpConversionPattern<mlir::torch::Torch::AtenIntBoolOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::AtenIntBoolOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value nativeBool = trident::torchext::GetOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(), adaptor.getA());
    mlir::Value nativeInt = mlir::arith::ExtUIOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(), nativeBool);
    rewriter.replaceOpWithNewOp<mlir::torch::TorchConversion::FromI64Op>(
        op, nativeInt);
    return mlir::success();
  }
};

class ConvertAtenSizeInt final
    : public mlir::OpConversionPattern<mlir::torch::Torch::AtenSizeIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::AtenSizeIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value nativeDim =
        trident::torchext::GetOp::create(
            rewriter, op.getLoc(), rewriter.getI64Type(), adaptor.getDim())
            .getResult();
    mlir::Value nativeSize = trident::tvm_ffi::TensorSizeOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(), adaptor.getSelf(),
        nativeDim);
    rewriter.replaceOpWithNewOp<mlir::torch::TorchConversion::FromI64Op>(
        op, nativeSize);
    return mlir::success();
  }
};

class ConvertAtenSubInt final
    : public mlir::OpConversionPattern<mlir::torch::Torch::AtenSubIntOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::AtenSubIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value nativeResult = mlir::arith::SubIOp::create(
        rewriter, op.getLoc(),
        trident::torchext::GetOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64Type(), adaptor.getA())
            .getResult(),
        trident::torchext::GetOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64Type(), adaptor.getB())
            .getResult());
    rewriter.replaceOpWithNewOp<mlir::torch::TorchConversion::FromI64Op>(
        op, nativeResult);
    return mlir::success();
  }
};

class GeneralizeAtenOpPattern final : public mlir::RewritePattern {
public:
  explicit GeneralizeAtenOpPattern(mlir::MLIRContext *context)
      : RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), /*benefit=*/1,
                       context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const final {
    const llvm::StringRef name = op->getName().getStringRef();
    if (!name.starts_with("torch.aten.")) {
      return mlir::failure();
    }
    mlir::NamedAttrList attributes(op->getAttrs());
    attributes.set("name", rewriter.getStringAttr(name));

    if (op->getNumRegions() != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected a regionless ATen operation");
    }
    rewriter.replaceOpWithNewOp<mlir::torch::Torch::OperatorOp>(
        op, op->getResultTypes(), op->getOperands(), attributes.getAttrs(),
        /*numRegions=*/0);
    return mlir::success();
  }
};

class GeneralizeAtenOpsPass final
    : public impl::GeneralizeAtenOpsBase<GeneralizeAtenOpsPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addIllegalOp<
        mlir::torch::Torch::AtenAddIntOp, mlir::torch::Torch::AtenFloordivIntOp,
        mlir::torch::Torch::AtenIntBoolOp, mlir::torch::Torch::AtenSizeIntOp,
        mlir::torch::Torch::AtenSubIntOp>();
    target.addLegalDialect<
        mlir::BuiltinDialect, mlir::func::FuncDialect,
        mlir::arith::ArithDialect, mlir::torch::Torch::TorchDialect,
        mlir::torch::TorchConversion::TorchConversionDialect,
        trident::tvm_ffi::TVMFFIDialect, trident::torchext::TorchExtDialect>();
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    mlir::RewritePatternSet specializedPatterns(&getContext());
    specializedPatterns
        .add<ConvertAtenAddInt, ConvertAtenFloordivInt, ConvertAtenIntBool,
             ConvertAtenSizeInt, ConvertAtenSubInt>(&getContext());
    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(specializedPatterns)))) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet generalPatterns(&getContext());
    generalPatterns.add<GeneralizeAtenOpPattern>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(generalPatterns));
  }
};

} // namespace trident::torch
