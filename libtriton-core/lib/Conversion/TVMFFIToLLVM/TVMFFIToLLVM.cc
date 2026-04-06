#include <cstdint>

#include "libtriton_core/Conversion/DLPackToLLVM/DLPackLLVMDescriptors.h"
#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFICAPIDescriptors.h"
#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFILLVMDescriptors.h"
#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tvm/ffi/c_api.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::tvm_ffi {
namespace {

constexpr std::int64_t kTVMFFIObjectHeaderBytes =
    static_cast<std::int64_t>(sizeof(TVMFFIObject));

mlir::Value materializeCast(mlir::OpBuilder &builder, mlir::Type resultType,
                            mlir::ValueRange inputs, mlir::Location loc) {
  if (inputs.size() != 1)
    return {};
  return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs)
      .getResult(0);
}

mlir::FailureOr<mlir::Value> emitTensorFromDLPackAsObjectHandle(
    mlir::ModuleOp moduleOp, mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value fromPtr, mlir::Value requireAlignment,
    mlir::Value requireContiguous) {
  mlir::MLIRContext *context = moduleOp.getContext();
  mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

  mlir::Value one =
      mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1LL).getResult();

  mlir::TypedValue<mlir::LLVM::LLVMPointerType> fromSlot =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, one)
              .getResult());
  mlir::LLVM::StoreOp::create(rewriter, loc, fromPtr, fromSlot);

  mlir::TypedValue<mlir::LLVM::LLVMPointerType> outSlot =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, one)
              .getResult());
  mlir::Value zeroPtr = mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy);
  mlir::LLVM::StoreOp::create(rewriter, loc, zeroPtr, outSlot);

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
      capi::getOrCreateTVMFFITensorFromDLPack(moduleOp);
  if (mlir::failed(calleeOrErr))
    return mlir::failure();

  mlir::LLVM::CallOp::create(
      rewriter, loc, *calleeOrErr,
      mlir::ValueRange{fromSlot, requireAlignment, requireContiguous, outSlot});

  return mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, outSlot).getResult();
}

struct LowerTensorFromDLPackOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::TensorFromDLPackOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::TensorFromDLPackOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    mlir::FailureOr<mlir::Value> objectHandle =
        emitTensorFromDLPackAsObjectHandle(
            moduleOp, rewriter, loc, adaptor.getFrom(),
            adaptor.getRequireAlignment(), adaptor.getRequireContiguous());
    if (mlir::failed(objectHandle))
      return mlir::failure();
    rewriter.replaceOp(op, *objectHandle);
    return mlir::success();
  }
};

struct LowerFromTensorOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::FromTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::FromTensorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);

    mlir::Type convertedAnyType =
        getTypeConverter()->convertType(op.getOutput().getType());
    mlir::LLVM::LLVMStructType anyLLVMType =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedAnyType);
    if (!anyLLVMType)
      return mlir::failure();

    mlir::TypedValue<mlir::IntegerType> typeIndexValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(
                rewriter, loc, i32Ty, static_cast<std::int64_t>(kTVMFFITensor))
                .getResult());
    mlir::TypedValue<mlir::IntegerType> zeroPaddingValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0LL)
                .getResult());
    mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                           adaptor.getInput())
                .getResult());

    libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
        libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::build(
            rewriter, loc, anyLLVMType, typeIndexValue, zeroPaddingValue,
            payloadBitsValue);

    rewriter.replaceOp(op, anyValue.as());
    return mlir::success();
  }
};

struct LowerToTensorOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::ToTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::ToTensorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    mlir::Type convertedTensorType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedTensorType)
      return mlir::failure();

    libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
        libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::from(
            adaptor.getInput());
    mlir::TypedValue<mlir::IntegerType> payloadBits =
        anyValue.payloadBits(rewriter, loc);
    mlir::Value payloadPtr =
        mlir::LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, payloadBits)
            .getResult();

    mlir::Value tensorCellPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i8Ty, payloadPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{kTVMFFIObjectHeaderBytes});

    mlir::Value tensorValue = mlir::LLVM::LoadOp::create(
        rewriter, loc, convertedTensorType, tensorCellPtr);
    rewriter.replaceOp(op, tensorValue);
    return mlir::success();
  }
};

class ConvertTVMFFIToLLVMPass
    : public mlir::PassWrapper<ConvertTVMFFIToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTVMFFIToLLVMPass)

  llvm::StringRef getArgument() const final {
    return "convert-tvm-ffi-to-llvm";
  }

  llvm::StringRef getDescription() const final {
    return "Lower TVMFFI dialect operations to LLVM dialect";
  }

  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static mlir::PassRegistration<ConvertTVMFFIToLLVMPass> kPass;

struct TVMFFIToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTVMFFIToLLVMTypeConversions(
    mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([](libtriton::dlpack::DLManagedTensorType type) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  typeConverter.addConversion([&](libtriton::dlpack::DLTensorType type)
                                  -> mlir::Type {
    return libtriton::conversion::utils::DLTensorLLVMDescriptor::getLLVMType(
        type.getContext(), typeConverter.getPointerBitwidth());
  });
  typeConverter.addConversion([](libtriton::tvm_ffi::AnyType type) {
    return libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  typeConverter.addConversion([](libtriton::tvm_ffi::ObjectHandleType type) {
    return libtriton::conversion::utils::TVMFFIObjectHandleLLVMDescriptor::
        getLLVMType(type.getContext());
  });
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);
}

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  populateTVMFFIToLLVMTypeConversions(typeConverter);

  patterns.add<LowerFromTensorOp, LowerTensorFromDLPackOp, LowerToTensorOp>(
      typeConverter, context);

  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect,
                         libtriton::tvm_ffi::TVMFFIDialect>();
  target.addIllegalOp<libtriton::tvm_ffi::FromTensorOp>();
  target.addIllegalOp<libtriton::tvm_ffi::TensorFromDLPackOp>();
  target.addIllegalOp<libtriton::tvm_ffi::ToTensorOp>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

std::unique_ptr<mlir::Pass> createConvertTVMFFIToLLVMPass() {
  return std::make_unique<ConvertTVMFFIToLLVMPass>();
}

void registerConvertTVMFFIToLLVMPass() {
  // Registration is handled by static PassRegistration above.
}

void registerTVMFFIToLLVMPasses() { registerConvertTVMFFIToLLVMPass(); }

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, libtriton::tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace libtriton::tvm_ffi
