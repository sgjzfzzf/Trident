#include "libtriton-core/Conversion/TVMFFIToLLVM/ToRules.h"

#include <cstdint>
#include <optional>

namespace libtriton::tvm_ffi::to::convert {

std::optional<mlir::Value>
buildAny(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
         const mlir::TypeConverter *typeConverter, mlir::Type anyType,
         mlir::TypedValue<mlir::IntegerType> typeIndexValue,
         mlir::TypedValue<mlir::IntegerType> payloadBitsValue) {
  const mlir::Type convertedAnyType = typeConverter->convertType(anyType);
  const mlir::LLVM::LLVMStructType anyLLVMType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedAnyType);
  if (!anyLLVMType) {
    return std::nullopt;
  }
  const mlir::TypedValue<mlir::IntegerType> zeroPaddingValue =
      emitIConstant<32>(rewriter, loc, rewriter.getContext(), 0);
  return conversion::utils::TVMFFIAnyLLVMDescriptor::build(
             rewriter, loc, anyLLVMType, typeIndexValue, zeroPaddingValue,
             payloadBitsValue)
      .as();
}

mlir::TypedValue<mlir::IntegerType>
castIntegerToI64(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 mlir::Type i64Ty,
                 mlir::TypedValue<mlir::IntegerType> integerValue) {
  const int32_t width = integerValue.getType().getWidth();
  if (width == 64) {
    return integerValue;
  }
  if (width < 64) {
    return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
        mlir::LLVM::ZExtOp::create(rewriter, loc, i64Ty, integerValue)
            .getResult());
  }
  return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
      mlir::LLVM::TruncOp::create(rewriter, loc, i64Ty, integerValue)
          .getResult());
}

mlir::Value
AnyToAnyConverter::convert(ToOp /*op*/, ToOp::Adaptor adaptor,
                           mlir::ConversionPatternRewriter & /*rewriter*/,
                           const mlir::TypeConverter * /*typeConverter*/) {
  return adaptor.getInput();
}

mlir::Value
ObjectHandleToAnyConverter::convert(ToOp op, ToOp::Adaptor adaptor,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  const mlir::TypedValue<mlir::IntegerType> tensorTypeIndexValue =
      emitIConstant<32>(rewriter, loc, context, kTVMFFITensor);
  const mlir::TypedValue<mlir::IntegerType> noneTypeIndexValue =
      emitIConstant<32>(rewriter, loc, context, kTVMFFINone);
  const mlir::TypedValue<mlir::LLVM::LLVMPointerType> nullHandleValue =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::ZeroOp::create(rewriter, loc,
                                     adaptor.getInput().getType())
              .getResult());
  const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
          mlir::LLVM::SelectOp::create(
              rewriter, loc, i32Ty,
              mlir::LLVM::ICmpOp::create(rewriter, loc,
                                         mlir::LLVM::ICmpPredicate::eq,
                                         adaptor.getInput(), nullHandleValue)
                  .getResult(),
              noneTypeIndexValue, tensorTypeIndexValue)
              .getResult());
  const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
          mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                         adaptor.getInput())
              .getResult());
  return *buildAny(rewriter, loc, typeConverter, op.getOutput().getType(),
                   typeIndexValue, payloadBitsValue);
}

mlir::Value
PointerToAnyConverter::convert(ToOp op, ToOp::Adaptor adaptor,
                               mlir::ConversionPatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
      emitIConstant<32>(rewriter, loc, context, kTVMFFIRawStr);
  const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
          mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty,
                                         adaptor.getInput())
              .getResult());
  return *buildAny(rewriter, loc, typeConverter, op.getOutput().getType(),
                   typeIndexValue, payloadBitsValue);
}

mlir::Value
Float64ToAnyConverter::convert(ToOp op, ToOp::Adaptor adaptor,
                               mlir::ConversionPatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
      emitIConstant<32>(rewriter, loc, context, kTVMFFIFloat);
  const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
          mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty,
                                        adaptor.getInput())
              .getResult());
  return *buildAny(rewriter, loc, typeConverter, op.getOutput().getType(),
                   typeIndexValue, payloadBitsValue);
}

mlir::Value
IntegerToAnyConverter::convert(ToOp op, ToOp::Adaptor adaptor,
                               mlir::ConversionPatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
      emitIConstant<32>(rewriter, loc, context, kTVMFFIInt);
  const mlir::TypedValue<mlir::IntegerType> payload = castIntegerToI64(
      rewriter, loc, i64Ty,
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(adaptor.getInput()));
  return *buildAny(rewriter, loc, typeConverter, op.getOutput().getType(),
                   typeIndexValue, payload);
}

mlir::Value
AnyToDLTensorConverter::convert(ToOp op, ToOp::Adaptor adaptor,
                                mlir::ConversionPatternRewriter &rewriter,
                                const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> anyTypeIndex =
      anyValue.typeIndex(rewriter, loc);
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(rewriter, loc);
  const mlir::Type convertedTensorType =
      typeConverter->convertType(op.getOutput().getType());
  const mlir::Value payloadPtr =
      mlir::LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, payloadBits)
          .getResult();
  return mlir::LLVM::LoadOp::create(
             rewriter, loc, convertedTensorType,
             mlir::LLVM::SelectOp::create(
                 rewriter, loc, ptrTy,
                 mlir::LLVM::ICmpOp::create(
                     rewriter, loc, mlir::LLVM::ICmpPredicate::eq, anyTypeIndex,
                     mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                                    kTVMFFITensor)
                         .getResult())
                     .getResult(),
                 mlir::LLVM::GEPOp::create(
                     rewriter, loc, ptrTy, i8Ty, payloadPtr,
                     llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)}),
                 payloadPtr))
      .getResult();
}

mlir::Value
AnyToFloat64Converter::convert(ToOp op, ToOp::Adaptor adaptor,
                               mlir::ConversionPatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(rewriter, loc);
  const mlir::Type convertedFloatType =
      typeConverter->convertType(op.getOutput().getType());
  return mlir::LLVM::BitcastOp::create(rewriter, loc, convertedFloatType,
                                       payloadBits)
      .getResult();
}

mlir::Value
AnyToObjectHandleConverter::convert(ToOp op, ToOp::Adaptor adaptor,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(rewriter, loc);
  const mlir::Type convertedPtrType =
      typeConverter->convertType(op.getOutput().getType());
  return mlir::LLVM::IntToPtrOp::create(rewriter, loc, convertedPtrType,
                                        payloadBits)
      .getResult();
}

mlir::Value
AnyToPointerConverter::convert(ToOp op, ToOp::Adaptor adaptor,
                               mlir::ConversionPatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(rewriter, loc);
  const mlir::Type convertedPtrType =
      typeConverter->convertType(op.getOutput().getType());
  return mlir::LLVM::IntToPtrOp::create(rewriter, loc, convertedPtrType,
                                        payloadBits)
      .getResult();
}

mlir::Value
AnyToIntegerConverter::convert(ToOp op, ToOp::Adaptor adaptor,
                               mlir::ConversionPatternRewriter &rewriter,
                               const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(rewriter, loc);
  const mlir::IntegerType targetIntegerType =
      mlir::dyn_cast<mlir::IntegerType>(op.getOutput().getType());
  const mlir::Type convertedIntegerType =
      typeConverter->convertType(op.getOutput().getType());
  const int32_t width = targetIntegerType.getWidth();
  if (width == 64) {
    return payloadBits;
  }
  if (width < 64) {
    return mlir::LLVM::TruncOp::create(rewriter, loc, convertedIntegerType,
                                       payloadBits)
        .getResult();
  }
  return mlir::LLVM::ZExtOp::create(rewriter, loc, convertedIntegerType,
                                    payloadBits)
      .getResult();
}

mlir::Value ObjectHandleToDLTensorConverter::convert(
    ToOp op, ToOp::Adaptor adaptor, mlir::ConversionPatternRewriter &rewriter,
    const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type convertedTensorType =
      typeConverter->convertType(op.getOutput().getType());
  const mlir::Value tensorPtr = mlir::LLVM::GEPOp::create(
      rewriter, loc, ptrTy, i8Ty, adaptor.getInput(),
      llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
  return mlir::LLVM::LoadOp::create(rewriter, loc, convertedTensorType,
                                    tensorPtr)
      .getResult();
}

} // namespace libtriton::tvm_ffi::to::convert
