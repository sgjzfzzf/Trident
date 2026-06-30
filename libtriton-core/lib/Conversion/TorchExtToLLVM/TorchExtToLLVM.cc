#include "libtriton-core/Conversion/TorchExtToLLVM/TorchExtToLLVM.h"
#include "libtriton-core/Conversion/Utils/TVMFFIUtils.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torchext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

/// Converts torchext.aoti.ListDeleteList to TVMFFIObjectDecRef() LLVM call.
class ConvertListDeleteListOp
    : public mlir::OpConversionPattern<ListDeleteListOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ListDeleteListOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = op.getContext();

    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return op.emitError("op is not inside a module");
    }

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        libtriton::conversion::utils::getOrCreateTVMFFIObjectDecRef(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }

    // The adapted list is a TVMFFIAny — extract the pointer from field[2].
    mlir::Value anyVal = adaptor.getList();
    mlir::Value payloadI64 = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, anyVal, llvm::ArrayRef<int64_t>{2});
    mlir::Value handle = mlir::LLVM::IntToPtrOp::create(
        rewriter, loc, mlir::LLVM::LLVMPointerType::get(ctx), payloadI64);
    mlir::LLVM::CallOp::create(rewriter, loc, *calleeOrErr,
                               mlir::ValueRange{handle});
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
  patterns.add<ConvertListDeleteListOp>(typeConverter, patterns.getContext());
  target.addIllegalOp<libtriton::torchext::ListDeleteListOp>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
}

void registerConvertTorchExtToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            libtriton::torchext::TorchExtDialect *dialect) {
    dialect->addInterfaces<TorchExtToLLVMDialectInterface>();
  });
}

} // namespace libtriton::torchext
