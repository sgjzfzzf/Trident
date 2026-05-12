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

mlir::FailureOr<mlir::TypedValue<mlir::IntegerType>>
castIntegerToI64Payload(mlir::ConversionPatternRewriter &rewriter,
                        mlir::Location loc, mlir::Type i64Ty,
                        mlir::TypedValue<mlir::IntegerType> integerValue) {
  const std::int32_t width = integerValue.getType().getWidth();
  if (width == 64) {
    return integerValue;
  } else if (width < 64) {
    return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
        mlir::LLVM::ZExtOp::create(rewriter, loc, i64Ty, integerValue)
            .getResult());
  } else {
    return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
        mlir::LLVM::TruncOp::create(rewriter, loc, i64Ty, integerValue)
            .getResult());
  }
}

mlir::FailureOr<mlir::Value> castI64PayloadToInteger(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    const mlir::TypeConverter *typeConverter, mlir::Type integerType,
    mlir::TypedValue<mlir::IntegerType> payloadBits) {
  const mlir::IntegerType targetIntegerType =
      mlir::dyn_cast<mlir::IntegerType>(integerType);
  if (!targetIntegerType) {
    return mlir::failure();
  }
  const mlir::Type convertedIntegerType =
      typeConverter->convertType(integerType);
  if (!convertedIntegerType) {
    return mlir::failure();
  }
  const std::int32_t width = targetIntegerType.getWidth();
  if (width == 64) {
    return payloadBits;
  } else if (width < 64) {
    return mlir::LLVM::TruncOp::create(rewriter, loc, convertedIntegerType,
                                       payloadBits)
        .getResult();
  } else {
    return mlir::LLVM::ZExtOp::create(rewriter, loc, convertedIntegerType,
                                      payloadBits)
        .getResult();
  }
}

mlir::FailureOr<mlir::Value> emitTensorFromDLPackAsObjectHandle(
    mlir::ModuleOp moduleOp, mlir::ConversionPatternRewriter &rewriter,
    mlir::Location loc, mlir::Value fromManaged, mlir::Value requireAlignment,
    mlir::Value requireContiguous) {
  mlir::MLIRContext *context = moduleOp.getContext();
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type managedTy = fromManaged.getType();
  const mlir::Value one =
      mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1LL).getResult();
  const mlir::FailureOr<mlir::LLVM::LLVMFuncOp> mallocOrErr =
      libtriton::conversion::utils::getOrCreateMalloc(moduleOp);
  if (mlir::failed(mallocOrErr)) {
    return mlir::failure();
  }
  const mlir::DataLayout layout(moduleOp);
  const std::optional<std::int64_t> managedSize =
      layout.getTypeSizeInBits(managedTy).getFixedValue();
  if (!managedSize.has_value() || *managedSize <= 0) {
    return mlir::failure();
  }
  const mlir::Value managedSizeBytes =
      mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, *managedSize / 8)
          .getResult();
  mlir::LLVM::CallOp managedAllocCall = mlir::LLVM::CallOp::create(
      rewriter, loc, *mallocOrErr, mlir::ValueRange{managedSizeBytes});
  const mlir::TypedValue<mlir::LLVM::LLVMPointerType> fromSlot =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          managedAllocCall.getResult());
  mlir::LLVM::StoreOp::create(rewriter, loc, fromManaged, fromSlot);
  const mlir::TypedValue<mlir::LLVM::LLVMPointerType> outSlot =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, one)
              .getResult());
  const mlir::Value zeroPtr = mlir::LLVM::ZeroOp::create(rewriter, loc, ptrTy);
  mlir::LLVM::StoreOp::create(rewriter, loc, zeroPtr, outSlot);
  const mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
      capi::getOrCreateTVMFFITensorFromDLPack(moduleOp);
  if (mlir::failed(calleeOrErr)) {
    return mlir::failure();
  }
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
    if (!moduleOp) {
      return mlir::failure();
    }
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<mlir::Value> objectHandle =
        emitTensorFromDLPackAsObjectHandle(
            moduleOp, rewriter, loc, adaptor.getFrom(),
            adaptor.getRequireAlignment(), adaptor.getRequireContiguous());
    if (mlir::failed(objectHandle)) {
      return mlir::failure();
    } else {
      rewriter.replaceOp(op, *objectHandle);
      return mlir::success();
    }
  }
};

struct LowerToOp : public mlir::OpConversionPattern<libtriton::tvm_ffi::ToOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::FailureOr<mlir::Value>
  lowerToAny(libtriton::tvm_ffi::ToOp op, OpAdaptor adaptor,
             mlir::ConversionPatternRewriter &rewriter) const {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    const mlir::Type inputType = op.getInput().getType();
    const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    const mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    if (mlir::isa<libtriton::tvm_ffi::AnyType>(inputType)) {
      return adaptor.getInput();
    } else if (mlir::isa<libtriton::tvm_ffi::ObjectHandleType>(inputType)) {
      const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
          emitI32Constant(rewriter, loc, context, kTVMFFITensor);
      const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
          mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
              mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                             adaptor.getInput())
                  .getResult());
      return buildAnyValue(rewriter, loc, getTypeConverter(),
                           op.getOutput().getType(), typeIndexValue,
                           payloadBitsValue);
    } else if (mlir::isa<mlir::LLVM::LLVMPointerType>(inputType)) {
      const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
          emitI32Constant(rewriter, loc, context, kTVMFFIRawStr);
      const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
          mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
              mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                             adaptor.getInput())
                  .getResult());
      return buildAnyValue(rewriter, loc, getTypeConverter(),
                           op.getOutput().getType(), typeIndexValue,
                           payloadBitsValue);
    } else if (mlir::isa<mlir::Float64Type>(inputType)) {
      const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
          emitI32Constant(rewriter, loc, context, kTVMFFIFloat);
      const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
          mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
              mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty,
                                            adaptor.getInput())
                  .getResult());
      return buildAnyValue(rewriter, loc, getTypeConverter(),
                           op.getOutput().getType(), typeIndexValue,
                           payloadBitsValue);
    } else if (mlir::isa<mlir::IntegerType>(inputType)) {
      const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
          emitI32Constant(rewriter, loc, context, kTVMFFIInt);
      const mlir::FailureOr<mlir::TypedValue<mlir::IntegerType>> payloadOrErr =
          castIntegerToI64Payload(
              rewriter, loc, i64Ty,
              mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
                  adaptor.getInput()));
      if (mlir::failed(payloadOrErr)) {
        return mlir::failure();
      }
      const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
          *payloadOrErr;
      return buildAnyValue(rewriter, loc, getTypeConverter(),
                           op.getOutput().getType(), typeIndexValue,
                           payloadBitsValue);
    } else {
      return mlir::failure();
    }
  }

  mlir::FailureOr<mlir::Value>
  lowerFromAny(libtriton::tvm_ffi::ToOp op, OpAdaptor adaptor,
               mlir::ConversionPatternRewriter &rewriter) const {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    const mlir::Type outputType = op.getOutput().getType();
    const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
    const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
    const libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
        libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::from(
            adaptor.getInput());
    const mlir::TypedValue<mlir::IntegerType> payloadBits =
        anyValue.payloadBits(rewriter, loc);
    if (mlir::isa<libtriton::tvm_ffi::AnyType>(outputType)) {
      return adaptor.getInput();
    } else if (mlir::isa<libtriton::dlpack::DLTensorType>(outputType)) {
      mlir::Type convertedTensorType =
          getTypeConverter()->convertType(outputType);
      if (!convertedTensorType)
        return mlir::failure();
      mlir::Value payloadPtr =
          mlir::LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, payloadBits)
              .getResult();
      mlir::Value tensorCellPtr = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrTy, i8Ty, payloadPtr,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{kTVMFFIObjectHeaderBytes});
      return mlir::LLVM::LoadOp::create(rewriter, loc, convertedTensorType,
                                        tensorCellPtr)
          .getResult();
    } else if (mlir::isa<mlir::Float64Type>(outputType)) {
      mlir::Type convertedFloatType =
          getTypeConverter()->convertType(outputType);
      if (!convertedFloatType)
        return mlir::failure();
      return mlir::LLVM::BitcastOp::create(rewriter, loc, convertedFloatType,
                                           payloadBits)
          .getResult();
    } else if (mlir::isa<mlir::LLVM::LLVMPointerType,
                         libtriton::tvm_ffi::ObjectHandleType>(outputType)) {
      mlir::Type convertedPtrType = getTypeConverter()->convertType(outputType);
      if (!convertedPtrType)
        return mlir::failure();
      return mlir::LLVM::IntToPtrOp::create(rewriter, loc, convertedPtrType,
                                            payloadBits)
          .getResult();
    } else if (mlir::isa<mlir::IntegerType>(outputType)) {
      return castI64PayloadToInteger(rewriter, loc, getTypeConverter(),
                                     outputType, payloadBits);
    } else {
      return mlir::failure();
    }
  }

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::ToOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Value> convertedValue = mlir::failure();
    if (mlir::isa<libtriton::tvm_ffi::AnyType>(op.getOutput().getType())) {
      convertedValue = lowerToAny(op, adaptor, rewriter);
    } else if (mlir::isa<libtriton::tvm_ffi::AnyType>(
                   op.getInput().getType())) {
      convertedValue = lowerFromAny(op, adaptor, rewriter);
    }
    if (mlir::failed(convertedValue)) {
      return mlir::failure();
    } else {
      rewriter.replaceOp(op, *convertedValue);
      return mlir::success();
    }
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
    if (!convertedAnyType || adaptor.getInput().getType() != convertedAnyType) {
      return mlir::failure();
    } else {
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    }
  }
};

struct LowerAnyToLLVMOp
    : public mlir::OpConversionPattern<libtriton::tvm_ffi::AnyToLLVMOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::AnyToLLVMOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getInput().getType() != op.getOutput().getType()) {
      return mlir::failure();
    } else {
      rewriter.replaceOp(op, adaptor.getInput());
      return mlir::success();
    }
  }
};

struct LowerErrorSetRaisedFromCStrOp
    : public mlir::OpConversionPattern<
          libtriton::tvm_ffi::ErrorSetRaisedFromCStrOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::tvm_ffi::ErrorSetRaisedFromCStrOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
      return mlir::failure();
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
        capi::getOrCreateTVMFFIErrorSetRaisedFromCStr(moduleOp);
    if (mlir::failed(calleeOrErr)) {
      return mlir::failure();
    }
    mlir::LLVM::CallOp::create(
        rewriter, op.getLoc(), *calleeOrErr,
        mlir::ValueRange{adaptor.getKind(), adaptor.getMessage()});
    rewriter.eraseOp(op);
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
  patterns.add<LowerToOp, LowerTensorFromDLPackOp, LowerAnyFromLLVMOp,
               LowerAnyToLLVMOp, LowerErrorSetRaisedFromCStrOp>(typeConverter,
                                                                context);
  target.addIllegalDialect<libtriton::tvm_ffi::TVMFFIDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, libtriton::tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace libtriton::tvm_ffi
