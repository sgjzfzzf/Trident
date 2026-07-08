//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "trident-core/Conversion/Utils/Check.h"
#include "trident-core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

namespace trident::torchext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOLLVM
#include "trident-core/Conversion/Passes.h.inc"

namespace {

/// Converts torchext.aoti.ObjectIncRef to TVMFFIObjectIncRef() LLVM call.
class ConvertObjectIncRefOp : public mlir::OpConversionPattern<ObjectIncRefOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ObjectIncRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = op.getContext();

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    mlir::LLVM::LLVMFuncOp callee = TRIDENT_CHECK_FAILURE(
        trident::conversion::utils::getOrCreateTVMFFIObjectIncRef(moduleOp));

    // The adapted object is a TVMFFIAny — extract the pointer from field[2].
    mlir::Value anyVal = adaptor.getObject();
    mlir::Value payloadI64 = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, anyVal, llvm::ArrayRef<int64_t>{2});
    mlir::Value handle = mlir::LLVM::IntToPtrOp::create(
        rewriter, loc, mlir::LLVM::LLVMPointerType::get(ctx), payloadI64);
    mlir::LLVM::CallOp::create(rewriter, loc, callee, mlir::ValueRange{handle});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// Converts torchext.aoti.ObjectDecRef to TVMFFIObjectDecRef() LLVM call.
class ConvertObjectDecRefOp : public mlir::OpConversionPattern<ObjectDecRefOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ObjectDecRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = op.getContext();

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    mlir::LLVM::LLVMFuncOp callee = TRIDENT_CHECK_FAILURE(
        trident::conversion::utils::getOrCreateTVMFFIObjectDecRef(moduleOp));

    // The adapted object is a TVMFFIAny — extract the pointer from field[2].
    mlir::Value anyVal = adaptor.getObject();
    mlir::Value payloadI64 = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, anyVal, llvm::ArrayRef<int64_t>{2});
    mlir::Value handle = mlir::LLVM::IntToPtrOp::create(
        rewriter, loc, mlir::LLVM::LLVMPointerType::get(ctx), payloadI64);
    mlir::LLVM::CallOp::create(rewriter, loc, callee, mlir::ValueRange{handle});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertTorchExtToLLVMPass
    : public impl::ConvertTorchExtToLLVMBase<ConvertTorchExtToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);
    torch::setupBackendTypeConversion(target, typeConverter);
    populateTorchExtToLLVMConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct TorchExtToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    // Setup type conversion for torch types before adding patterns, so that
    // the type converter can handle Torch tensor/bool/int/float/optional etc.
    // types when patterns query adaptor types.
    torch::setupBackendTypeConversion(target, typeConverter);
    populateTorchExtToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTorchExtToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertObjectIncRefOp, ConvertObjectDecRefOp>(
      typeConverter, patterns.getContext());
  target.addIllegalOp<trident::torchext::ObjectIncRefOp,
                      trident::torchext::ObjectDecRefOp>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
}

void registerConvertTorchExtToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, trident::torchext::TorchExtDialect *dialect) {
        dialect->addInterfaces<TorchExtToLLVMDialectInterface>();
      });
}

} // namespace trident::torchext
