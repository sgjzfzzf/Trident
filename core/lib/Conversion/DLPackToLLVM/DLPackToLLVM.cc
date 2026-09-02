//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTDLPACKTOLLVM
#include "trident/core/Conversion/Passes.h.inc"

namespace {

class ConvertTensorDeviceOp final
    : public mlir::OpConversionPattern<dlpack::TensorDeviceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(dlpack::TensorDeviceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::Value const device = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc,
        mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {rewriter.getI32Type(), rewriter.getI32Type()}),
        adaptor.getTensor(), llvm::ArrayRef<int64_t>{1});
    mlir::Value const deviceType =
        mlir::LLVM::ExtractValueOp::create(rewriter, loc, rewriter.getI32Type(),
                                           device, llvm::ArrayRef<int64_t>{0});
    mlir::Value const deviceIndex =
        mlir::LLVM::ExtractValueOp::create(rewriter, loc, rewriter.getI32Type(),
                                           device, llvm::ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, {deviceType, deviceIndex});
    return mlir::success();
  }
};

class ConvertTensorDimOp final
    : public mlir::OpConversionPattern<dlpack::TensorDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(dlpack::TensorDimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const ndim = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), adaptor.getTensor(),
        llvm::ArrayRef<int64_t>{2});
    rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(op, rewriter.getI64Type(),
                                                    ndim);
    return mlir::success();
  }
};

class ConvertTensorDataOp final
    : public mlir::OpConversionPattern<dlpack::TensorDataOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(dlpack::TensorDataOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        adaptor.getTensor(), llvm::ArrayRef<int64_t>{0});
    return mlir::success();
  }
};

class ConvertTensorDTypeOp final
    : public mlir::OpConversionPattern<dlpack::TensorDTypeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(dlpack::TensorDTypeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const dtype = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(),
        mlir::LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(), {rewriter.getI8Type(), rewriter.getI8Type(),
                                    rewriter.getI16Type()}),
        adaptor.getTensor(), llvm::ArrayRef<int64_t>{3});
    mlir::Value const code = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), rewriter.getI8Type(), dtype,
        llvm::ArrayRef<int64_t>{0});
    mlir::Value const bits = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), rewriter.getI8Type(), dtype,
        llvm::ArrayRef<int64_t>{1});
    mlir::Value const lanes = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), rewriter.getI16Type(), dtype,
        llvm::ArrayRef<int64_t>{2});
    rewriter.replaceOp(op, {code, bits, lanes});
    return mlir::success();
  }
};

template <typename SourceOp, int64_t Field>
class ConvertTensorIndexedMetadataOp final
    : public mlir::OpConversionPattern<SourceOp> {
public:
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const values = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(),
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        adaptor.getTensor(), llvm::ArrayRef<int64_t>{Field});
    mlir::Value const element = mlir::LLVM::GEPOp::create(
        rewriter, op.getLoc(),
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext()),
        rewriter.getI64Type(), values,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{adaptor.getIndex()});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, rewriter.getI64Type(),
                                                    element);
    return mlir::success();
  }
};

class ConvertTensorStorageOffsetOp final
    : public mlir::OpConversionPattern<dlpack::TensorStorageOffsetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(dlpack::TensorStorageOffsetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const offset = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(), adaptor.getTensor(),
        llvm::ArrayRef<int64_t>{6});
    rewriter.replaceOp(op, offset);
    return mlir::success();
  }
};

} // namespace

void populateDLPackToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  typeConverter.addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
    if (mlir::isa<dlpack::DLTensorType>(type)) {
      return dlpack::DLTensorType::getLLVMType(type.getContext());
    }
    return std::nullopt;
  });
  patterns.add<ConvertTensorDataOp, ConvertTensorDeviceOp, ConvertTensorDimOp,
               ConvertTensorDTypeOp,
               ConvertTensorIndexedMetadataOp<dlpack::TensorSizeOp, 4>,
               ConvertTensorStorageOffsetOp,
               ConvertTensorIndexedMetadataOp<dlpack::TensorStrideOp, 5>>(
      typeConverter, patterns.getContext());
  target.addIllegalDialect<dlpack::DLPackDialect>();
}

class DLPackToLLVMDialectInterface final
    : public mlir::ConvertToLLVMPatternInterface {
public:
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateDLPackToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

void registerConvertDLPackToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *, dlpack::DLPackDialect *dialect) {
        dialect->addInterfaces<DLPackToLLVMDialectInterface>();
      });
}

class ConvertDLPackToLLVMPass final
    : public impl::ConvertDLPackToLLVMBase<ConvertDLPackToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::RewritePatternSet patterns(&getContext());
    populateDLPackToLLVMConversionPatterns(target, typeConverter, patterns);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalDialect<mlir::BuiltinDialect, mlir::func::FuncDialect,
                           mlir::LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return mlir::isLegalForReturnOpTypeConversionPattern(op,
                                                               typeConverter);
        });
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::conversion
