#include "libtriton-core/Conversion/TVMFFIToLLVM/ToConvertPatterns.h"

#include <cstdint>
#include <optional>

namespace libtriton::tvm_ffi {

std::optional<mlir::Value>
buildAny(mlir::OpBuilder &builder, mlir::Location loc,
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
      conversion::utils::emitI32Constant(builder, loc, builder.getContext(), 0);
  return conversion::utils::TVMFFIAnyLLVMDescriptor::build(
             builder, loc, anyLLVMType, typeIndexValue, zeroPaddingValue,
             payloadBitsValue)
      .as();
}

std::optional<mlir::Value>
toAnyStruct(mlir::Value adaptedInput, mlir::Type originalInputType,
            mlir::OpBuilder &builder, mlir::Location loc,
            const mlir::TypeConverter *typeConverter) {
  mlir::MLIRContext *context = builder.getContext();
  const mlir::Type anyType = AnyType::get(context);
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);

  // AnyType → already the LLVM struct; return as-is.
  if (mlir::isa<AnyType>(originalInputType)) {
    return adaptedInput;
  }

  // ObjectHandleType → ptr-to-any: null check → kTVMFFINone or kTVMFFITensor.
  if (mlir::isa<ObjectHandleType>(originalInputType)) {
    const mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    const mlir::TypedValue<mlir::IntegerType> tensorTypeIndexValue =
        conversion::utils::emitI32Constant(builder, loc, context,
                                           kTVMFFITensor);
    const mlir::TypedValue<mlir::IntegerType> noneTypeIndexValue =
        conversion::utils::emitI32Constant(builder, loc, context, kTVMFFINone);
    const mlir::TypedValue<mlir::LLVM::LLVMPointerType> nullHandleValue =
        mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
            mlir::LLVM::ZeroOp::create(builder, loc, adaptedInput.getType())
                .getResult());
    const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::SelectOp::create(
                builder, loc, i32Ty,
                mlir::LLVM::ICmpOp::create(builder, loc,
                                           mlir::LLVM::ICmpPredicate::eq,
                                           adaptedInput, nullHandleValue)
                    .getResult(),
                noneTypeIndexValue, tensorTypeIndexValue)
                .getResult());
    const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, adaptedInput)
                .getResult());
    return buildAny(builder, loc, typeConverter, anyType, typeIndexValue,
                    payloadBitsValue);
  }

  // LLVMPointerType → kTVMFFIRawStr.
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(originalInputType)) {
    const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
        conversion::utils::emitI32Constant(builder, loc, context,
                                           kTVMFFIRawStr);
    const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, adaptedInput)
                .getResult());
    return buildAny(builder, loc, typeConverter, anyType, typeIndexValue,
                    payloadBitsValue);
  }

  // Float64Type → kTVMFFIFloat.
  if (mlir::isa<mlir::Float64Type>(originalInputType)) {
    const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
        conversion::utils::emitI32Constant(builder, loc, context, kTVMFFIFloat);
    const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::BitcastOp::create(builder, loc, i64Ty, adaptedInput)
                .getResult());
    return buildAny(builder, loc, typeConverter, anyType, typeIndexValue,
                    payloadBitsValue);
  }

  // IntegerType → kTVMFFIInt.
  if (mlir::isa<mlir::IntegerType>(originalInputType)) {
    const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
        conversion::utils::emitI32Constant(builder, loc, context, kTVMFFIInt);
    const mlir::TypedValue<mlir::IntegerType> payload =
        conversion::utils::castIntegerTo(
            builder, loc, mlir::IntegerType::get(context, 64),
            mlir::cast<mlir::TypedValue<mlir::IntegerType>>(adaptedInput));
    return buildAny(builder, loc, typeConverter, anyType, typeIndexValue,
                    payload);
  }

  return std::nullopt;
}

mlir::Value
AnyToAnyPattern::convert(ToOp /*op*/, ToOp::Adaptor adaptor,
                         mlir::OpBuilder & /*builder*/,
                         const mlir::TypeConverter * /*typeConverter*/) {
  return adaptor.getInput();
}

mlir::Value
AnyToDLTensorPattern::convert(ToOp op, ToOp::Adaptor adaptor,
                              mlir::OpBuilder &builder,
                              const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> anyTypeIndex =
      anyValue.typeIndex(builder, loc);
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(builder, loc);
  const mlir::Type convertedTensorType =
      typeConverter->convertType(op.getOutput().getType());
  const mlir::Value payloadPtr =
      mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, payloadBits)
          .getResult();
  return mlir::LLVM::LoadOp::create(
             builder, loc, convertedTensorType,
             mlir::LLVM::SelectOp::create(
                 builder, loc, ptrTy,
                 mlir::LLVM::ICmpOp::create(
                     builder, loc, mlir::LLVM::ICmpPredicate::eq, anyTypeIndex,
                     mlir::LLVM::ConstantOp::create(builder, loc, i32Ty,
                                                    kTVMFFITensor)
                         .getResult())
                     .getResult(),
                 mlir::LLVM::GEPOp::create(
                     builder, loc, ptrTy, i8Ty, payloadPtr,
                     llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)}),
                 payloadPtr))
      .getResult();
}

mlir::Value
AnyToFloat64Pattern::convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(builder, loc);
  const mlir::Type convertedFloatType =
      typeConverter->convertType(op.getOutput().getType());
  return mlir::LLVM::BitcastOp::create(builder, loc, convertedFloatType,
                                       payloadBits)
      .getResult();
}

mlir::Value
AnyToIntegerPattern::convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(builder, loc);
  const mlir::IntegerType targetIntegerType =
      mlir::dyn_cast<mlir::IntegerType>(op.getOutput().getType());
  const mlir::Type convertedIntegerType =
      typeConverter->convertType(op.getOutput().getType());
  const int32_t width = targetIntegerType.getWidth();
  if (width == 64) {
    return payloadBits;
  }
  if (width < 64) {
    return mlir::LLVM::TruncOp::create(builder, loc, convertedIntegerType,
                                       payloadBits)
        .getResult();
  }
  return mlir::LLVM::ZExtOp::create(builder, loc, convertedIntegerType,
                                    payloadBits)
      .getResult();
}

mlir::Value
AnyToObjectHandlePattern::convert(ToOp op, ToOp::Adaptor adaptor,
                                  mlir::OpBuilder &builder,
                                  const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(builder, loc);
  const mlir::Type convertedPtrType =
      typeConverter->convertType(op.getOutput().getType());
  return mlir::LLVM::IntToPtrOp::create(builder, loc, convertedPtrType,
                                        payloadBits)
      .getResult();
}

mlir::Value
AnyToPointerPattern::convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  const conversion::utils::TVMFFIAnyLLVMDescriptor anyValue =
      conversion::utils::TVMFFIAnyLLVMDescriptor::from(adaptor.getInput());
  const mlir::TypedValue<mlir::IntegerType> payloadBits =
      anyValue.payloadBits(builder, loc);
  const mlir::Type convertedPtrType =
      typeConverter->convertType(op.getOutput().getType());
  return mlir::LLVM::IntToPtrOp::create(builder, loc, convertedPtrType,
                                        payloadBits)
      .getResult();
}

mlir::Value
Float64ToAnyPattern::convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
      conversion::utils::emitI32Constant(builder, loc, context, kTVMFFIFloat);
  const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
          mlir::LLVM::BitcastOp::create(builder, loc, i64Ty, adaptor.getInput())
              .getResult());
  return *buildAny(builder, loc, typeConverter, op.getOutput().getType(),
                   typeIndexValue, payloadBitsValue);
}

mlir::Value
IntegerToAnyPattern::convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::IntegerType i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
      conversion::utils::emitI32Constant(builder, loc, context, kTVMFFIInt);
  const mlir::TypedValue<mlir::IntegerType> payload =
      conversion::utils::castIntegerTo(
          builder, loc, i64Ty,
          mlir::cast<mlir::TypedValue<mlir::IntegerType>>(adaptor.getInput()));
  return *buildAny(builder, loc, typeConverter, op.getOutput().getType(),
                   typeIndexValue, payload);
}

mlir::Value
ObjectHandleToAnyPattern::convert(ToOp op, ToOp::Adaptor adaptor,
                                  mlir::OpBuilder &builder,
                                  const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  const mlir::TypedValue<mlir::IntegerType> tensorTypeIndexValue =
      conversion::utils::emitI32Constant(builder, loc, context, kTVMFFITensor);
  const mlir::TypedValue<mlir::IntegerType> noneTypeIndexValue =
      conversion::utils::emitI32Constant(builder, loc, context, kTVMFFINone);
  const mlir::TypedValue<mlir::LLVM::LLVMPointerType> nullHandleValue =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::ZeroOp::create(builder, loc, adaptor.getInput().getType())
              .getResult());
  const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
          mlir::LLVM::SelectOp::create(
              builder, loc, i32Ty,
              mlir::LLVM::ICmpOp::create(builder, loc,
                                         mlir::LLVM::ICmpPredicate::eq,
                                         adaptor.getInput(), nullHandleValue)
                  .getResult(),
              noneTypeIndexValue, tensorTypeIndexValue)
              .getResult());
  const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
          mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty,
                                         adaptor.getInput())
              .getResult());
  return *buildAny(builder, loc, typeConverter, op.getOutput().getType(),
                   typeIndexValue, payloadBitsValue);
}

mlir::Value ObjectHandleToDLTensorPattern::convert(
    ToOp op, ToOp::Adaptor adaptor, mlir::OpBuilder &builder,
    const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type convertedTensorType =
      typeConverter->convertType(op.getOutput().getType());
  const mlir::Value tensorPtr = mlir::LLVM::GEPOp::create(
      builder, loc, ptrTy, i8Ty, adaptor.getInput(),
      llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
  return mlir::LLVM::LoadOp::create(builder, loc, convertedTensorType,
                                    tensorPtr)
      .getResult();
}

mlir::Value
PointerToAnyPattern::convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter) {
  const mlir::Location loc = op.getLoc();
  mlir::MLIRContext *context = op.getContext();
  const mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
  const mlir::TypedValue<mlir::IntegerType> typeIndexValue =
      conversion::utils::emitI32Constant(builder, loc, context, kTVMFFIRawStr);
  const mlir::TypedValue<mlir::IntegerType> payloadBitsValue =
      mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
          mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty,
                                         adaptor.getInput())
              .getResult());
  return *buildAny(builder, loc, typeConverter, op.getOutput().getType(),
                   typeIndexValue, payloadBitsValue);
}

} // namespace libtriton::tvm_ffi
