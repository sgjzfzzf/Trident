#include <cstdint>

#include "libtriton-core/Conversion/DLPackToLLVM/DLPackLLVMDescriptors.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFICAPIDescriptors.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFILLVMDescriptors.h"
#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/Utils/StdLibCFunctionDeclUtils.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackOps.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tvm/ffi/c_api.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

constexpr std::int64_t kTVMFFIObjectHeaderBytes = sizeof(TVMFFIObject);

mlir::TypedValue<mlir::IntegerType>
emitI32Constant(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                mlir::MLIRContext *context, std::int64_t value) {
  mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
      mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, value).getResult());
}

mlir::FailureOr<mlir::LLVM::LLVMStructType>
getConvertedAnyLLVMType(const mlir::TypeConverter *typeConverter,
                        mlir::Type anyType) {
  mlir::Type convertedAnyType = typeConverter->convertType(anyType);
  mlir::LLVM::LLVMStructType anyLLVMType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedAnyType);
  if (!anyLLVMType)
    return mlir::failure();
  return anyLLVMType;
}

mlir::FailureOr<mlir::Value>
buildAnyValue(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              const mlir::TypeConverter *typeConverter, mlir::Type anyType,
              mlir::TypedValue<mlir::IntegerType> typeIndexValue,
              mlir::TypedValue<mlir::IntegerType> payloadBitsValue) {
  mlir::FailureOr<mlir::LLVM::LLVMStructType> anyLLVMType =
      getConvertedAnyLLVMType(typeConverter, anyType);
  if (mlir::failed(anyLLVMType))
    return mlir::failure();

  mlir::TypedValue<mlir::IntegerType> zeroPaddingValue =
      emitI32Constant(rewriter, loc, rewriter.getContext(), 0LL);

  return libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::build(
             rewriter, loc, *anyLLVMType, typeIndexValue, zeroPaddingValue,
             payloadBitsValue)
      .as();
}

mlir::FailureOr<mlir::Value> emitTensorFromDLPackAsObjectHandle(
    mlir::ModuleOp moduleOp, mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value fromManaged, mlir::Value requireAlignment,
    mlir::Value requireContiguous) {
  mlir::MLIRContext *context = moduleOp.getContext();
  mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type managedTy = fromManaged.getType();

  mlir::Value one =
      mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1LL).getResult();

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> mallocOrErr =
      libtriton::conversion::utils::getOrCreateMalloc(moduleOp);
  if (mlir::failed(mallocOrErr))
    return mlir::failure();

  mlir::DataLayout layout(moduleOp);
  std::optional<std::int64_t> managedSize =
      layout.getTypeSizeInBits(managedTy).getFixedValue();
  if (!managedSize.has_value() || *managedSize <= 0)
    return mlir::failure();

  mlir::Value managedSizeBytes =
      mlir::LLVM::ConstantOp::create(
          rewriter, loc, i64Ty, static_cast<std::int64_t>(*managedSize / 8))
          .getResult();
  mlir::LLVM::CallOp managedAllocCall = mlir::LLVM::CallOp::create(
      rewriter, loc, *mallocOrErr, mlir::ValueRange{managedSizeBytes});
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> fromSlot =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          managedAllocCall.getResult());

  mlir::LLVM::StoreOp::create(rewriter, loc, fromManaged, fromSlot);

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
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);

    mlir::TypedValue<mlir::IntegerType> typeIndexValue = emitI32Constant(
        rewriter, loc, context, static_cast<std::int64_t>(kTVMFFITensor));
    mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                           adaptor.getInput())
                .getResult());

    mlir::FailureOr<mlir::Value> anyValue = buildAnyValue(
        rewriter, loc, getTypeConverter(), op.getOutput().getType(),
        typeIndexValue, payloadBitsValue);
    if (mlir::failed(anyValue))
      return mlir::failure();

    rewriter.replaceOp(op, *anyValue);
    return mlir::success();
  }
};

struct LowerFromIntOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::FromIntOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::FromIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    mlir::TypedValue<mlir::IntegerType> typeIndexValue = emitI32Constant(
        rewriter, loc, context, static_cast<std::int64_t>(kTVMFFIInt));
    mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(adaptor.getInput());

    mlir::FailureOr<mlir::Value> anyValue = buildAnyValue(
        rewriter, loc, getTypeConverter(), op.getOutput().getType(),
        typeIndexValue, payloadBitsValue);
    if (mlir::failed(anyValue))
      return mlir::failure();

    rewriter.replaceOp(op, *anyValue);
    return mlir::success();
  }
};

struct LowerToIntOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::ToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::ToIntOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
        libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::from(
            adaptor.getInput());
    rewriter.replaceOp(op, anyValue.payloadBits(rewriter, loc));
    return mlir::success();
  }
};

struct LowerFromFloatOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::FromFloatOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::FromFloatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);

    mlir::TypedValue<mlir::IntegerType> typeIndexValue = emitI32Constant(
        rewriter, loc, context, static_cast<std::int64_t>(kTVMFFIFloat));
    mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty,
                                          adaptor.getInput())
                .getResult());

    mlir::FailureOr<mlir::Value> anyValue = buildAnyValue(
        rewriter, loc, getTypeConverter(), op.getOutput().getType(),
        typeIndexValue, payloadBitsValue);
    if (mlir::failed(anyValue))
      return mlir::failure();

    rewriter.replaceOp(op, *anyValue);
    return mlir::success();
  }
};

struct LowerToFloatOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::ToFloatOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::ToFloatOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::Type convertedFloatType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedFloatType)
      return mlir::failure();

    libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
        libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::from(
            adaptor.getInput());
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(
        op, convertedFloatType, anyValue.payloadBits(rewriter, loc));
    return mlir::success();
  }
};

struct LowerFromStrOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::FromStrOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::FromStrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);

    mlir::TypedValue<mlir::IntegerType> typeIndexValue = emitI32Constant(
        rewriter, loc, context, static_cast<std::int64_t>(kTVMFFIRawStr));
    mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                           adaptor.getInput())
                .getResult());

    mlir::FailureOr<mlir::Value> anyValue = buildAnyValue(
        rewriter, loc, getTypeConverter(), op.getOutput().getType(),
        typeIndexValue, payloadBitsValue);
    if (mlir::failed(anyValue))
      return mlir::failure();

    rewriter.replaceOp(op, *anyValue);
    return mlir::success();
  }
};

struct LowerToStrOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::ToStrOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::ToStrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::Type convertedPtrType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedPtrType)
      return mlir::failure();

    libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
        libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::from(
            adaptor.getInput());
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(
        op, convertedPtrType, anyValue.payloadBits(rewriter, loc));
    return mlir::success();
  }
};

struct LowerFromObjectOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::FromObjectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::FromObjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Type i32Ty = mlir::IntegerType::get(context, 32);

    mlir::TypedValue<mlir::IntegerType> typeIndexValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::LoadOp::create(rewriter, loc, i32Ty, adaptor.getInput())
                .getResult());
    mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                           adaptor.getInput())
                .getResult());

    mlir::FailureOr<mlir::Value> anyValue = buildAnyValue(
        rewriter, loc, getTypeConverter(), op.getOutput().getType(),
        typeIndexValue, payloadBitsValue);
    if (mlir::failed(anyValue))
      return mlir::failure();

    rewriter.replaceOp(op, *anyValue);
    return mlir::success();
  }
};

struct LowerToObjectOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::ToObjectOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::ToObjectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::Type convertedPtrType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedPtrType)
      return mlir::failure();

    libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
        libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::from(
            adaptor.getInput());
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(
        op, convertedPtrType, anyValue.payloadBits(rewriter, loc));
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

struct LowerAnyFromLLVMOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::AnyFromLLVMOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::AnyFromLLVMOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type convertedAnyType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedAnyType || adaptor.getInput().getType() != convertedAnyType)
      return mlir::failure();
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

struct LowerAnyToLLVMOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::AnyToLLVMOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::AnyToLLVMOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getInput().getType() != op.getOutput().getType())
      return mlir::failure();
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

class ConvertTVMFFIToLLVMPass
    : public impl::ConvertTVMFFIToLLVMBase<ConvertTVMFFIToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addIllegalDialect<libtriton::tvm_ffi::TVMFFIDialect>();
    target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::Operation *op) {
          return mlir::isLegalForReturnOpTypeConversionPattern(op,
                                                               typeConverter);
        });
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *op) {
      return !llvm::isa<libtriton::tvm_ffi::TVMFFIDialect>(op->getDialect());
    });

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

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

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  typeConverter.addConversion(
      [&](libtriton::dlpack::DLManagedTensorType type) -> mlir::Type {
        return libtriton::conversion::utils::DLManagedTensorLLVMDescriptor::
            getLLVMType(type.getContext(), typeConverter.getPointerBitwidth());
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
  patterns.add<LowerFromFloatOp, LowerFromIntOp, LowerFromObjectOp,
               LowerFromStrOp, LowerFromTensorOp, LowerTensorFromDLPackOp,
               LowerToFloatOp, LowerToIntOp, LowerToObjectOp, LowerToStrOp,
               LowerToTensorOp, LowerAnyFromLLVMOp, LowerAnyToLLVMOp>(
      typeConverter, context);

  target.addIllegalDialect<libtriton::tvm_ffi::TVMFFIDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, libtriton::tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace libtriton::tvm_ffi
