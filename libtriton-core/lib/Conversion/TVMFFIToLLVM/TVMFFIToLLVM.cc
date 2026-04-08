#include <cstdint>

#include "libtriton_core/Analysis/MemRefOriginAnalysis/MemRefOriginAnalysis.h"
#include "libtriton_core/Conversion/DLPackToLLVM/DLPackLLVMDescriptors.h"
#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFICAPIDescriptors.h"
#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFILLVMDescriptors.h"
#include "libtriton_core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton_core/Conversion/Utils/StdLibCFunctionDeclUtils.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackOps.h"
#include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton_core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tvm/ffi/c_api.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#define GEN_PASS_DEF_EMITTVMFFIINTERFACE
#include "libtriton_core/Conversion/TVMFFIToLLVM/Passes.h.inc"

namespace {

constexpr std::int64_t kTVMFFIObjectHeaderBytes =
    static_cast<std::int64_t>(sizeof(TVMFFIObject));
constexpr llvm::StringLiteral kTVMFFIInterfaceAttr =
    "tvm_ffi.emit_tvm_ffi_interface";
constexpr llvm::StringLiteral kTVMFFIWrapperPrefix = "__tvm_ffi_";

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

mlir::Value emitI32Constant(mlir::OpBuilder &builder, mlir::Location loc,
                            std::int32_t value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value, 32)
      .getResult();
}

mlir::Value emitI64Constant(mlir::OpBuilder &builder, mlir::Location loc,
                            std::int64_t value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
      .getResult();
}

mlir::LLVM::LLVMPointerType getOpaquePtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(context);
}

mlir::Value emitAnyBufferSlotPtr(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value bufferPtr,
                                 mlir::TypedValue<mlir::IntegerType> index,
                                 mlir::Type anyType) {
  mlir::Type ptrTy = getOpaquePtrType(builder.getContext());
  return mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyType, bufferPtr,
                                   llvm::ArrayRef<mlir::LLVM::GEPArg>{index})
      .getResult();
}

mlir::FailureOr<mlir::Value> emitUnboxAnyValue(mlir::OpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::Type valueType,
                                               mlir::Value anyValue) {
  mlir::MLIRContext *context = builder.getContext();
  if (mlir::isa<libtriton::tvm_ffi::AnyType>(valueType))
    return anyValue;
  if (mlir::isa<libtriton::tvm_ffi::ObjectHandleType>(valueType)) {
    return libtriton::tvm_ffi::ToObjectOp::create(builder, loc, valueType,
                                                  anyValue)
        .getOutput();
  }
  if (mlir::isa<libtriton::dlpack::DLTensorType>(valueType)) {
    return libtriton::tvm_ffi::ToTensorOp::create(builder, loc, valueType,
                                                  anyValue)
        .getOutput();
  }
  if (mlir::isa<mlir::BaseMemRefType>(valueType)) {
    mlir::Type tensorType = libtriton::dlpack::DLTensorType::get(context);
    mlir::Value tensor = libtriton::tvm_ffi::ToTensorOp::create(
                             builder, loc, tensorType, anyValue)
                             .getOutput();
    return libtriton::dlpack::ToMemRefOp::create(builder, loc, valueType,
                                                 tensor)
        .getOutput();
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(valueType)) {
    return libtriton::tvm_ffi::ToStrOp::create(builder, loc, valueType,
                                               anyValue)
        .getOutput();
  }
  if (mlir::isa<mlir::Float64Type>(valueType)) {
    return libtriton::tvm_ffi::ToFloatOp::create(builder, loc, valueType,
                                                 anyValue)
        .getOutput();
  }
  mlir::IntegerType integerType = mlir::dyn_cast<mlir::IntegerType>(valueType);
  if (integerType && integerType.getWidth() == 64) {
    return libtriton::tvm_ffi::ToIntOp::create(builder, loc, valueType,
                                               anyValue)
        .getOutput();
  }
  mlir::emitError(loc) << "unsupported TVM FFI wrapper parameter type: "
                       << valueType;
  return mlir::failure();
}

mlir::FailureOr<mlir::Value> emitBoxAnyValue(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::DataFlowSolver &solver,
                                             mlir::Value value) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::Type valueType = value.getType();
  mlir::Type anyType = libtriton::tvm_ffi::AnyType::get(context);
  if (mlir::isa<libtriton::tvm_ffi::AnyType>(valueType))
    return value;
  if (mlir::isa<libtriton::tvm_ffi::ObjectHandleType>(valueType)) {
    return libtriton::tvm_ffi::FromObjectOp::create(builder, loc, anyType,
                                                    value)
        .getOutput();
  }
  if (mlir::isa<libtriton::dlpack::DLTensorType>(valueType)) {
    return libtriton::tvm_ffi::FromTensorOp::create(builder, loc, anyType,
                                                    value)
        .getOutput();
  }
  if (mlir::isa<mlir::BaseMemRefType>(valueType)) {
    mlir::Type managedType =
        libtriton::dlpack::DLManagedTensorType::get(context);
    mlir::Type objectHandleType =
        libtriton::tvm_ffi::ObjectHandleType::get(context);
    const libtriton::analysis::MemRefOriginKind origin =
        libtriton::analysis::resolveMemRefOrigin(solver, value);

    // TVMFFITensorFromDLPack currently accepts !dlpack.managed_tensor, so use
    // FromMemRefOwnedOp for all origins and keep the origin query for future
    // policy extension.
    (void)origin;
    mlir::Value managed = libtriton::dlpack::FromMemRefOwnedOp::create(
                              builder, loc, managedType, value)
                              .getOutput();

    mlir::Value zero =
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 32).getResult();
    mlir::Value handle =
        libtriton::tvm_ffi::TensorFromDLPackOp::create(
            builder, loc, objectHandleType, managed, zero, zero)
            .getOutput();
    return libtriton::tvm_ffi::FromTensorOp::create(builder, loc, anyType,
                                                    handle)
        .getOutput();
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(valueType)) {
    return libtriton::tvm_ffi::FromStrOp::create(builder, loc, anyType, value)
        .getOutput();
  }
  if (mlir::isa<mlir::Float64Type>(valueType)) {
    return libtriton::tvm_ffi::FromFloatOp::create(builder, loc, anyType, value)
        .getOutput();
  }
  mlir::IntegerType integerType = mlir::dyn_cast<mlir::IntegerType>(valueType);
  if (integerType && integerType.getWidth() == 64) {
    return libtriton::tvm_ffi::FromIntOp::create(builder, loc, anyType, value)
        .getOutput();
  }
  mlir::emitError(loc) << "unsupported TVM FFI wrapper return type: "
                       << valueType;
  return mlir::failure();
}

mlir::FailureOr<mlir::func::FuncOp>
buildEmitTVMFFIInterfaceWrapper(mlir::ModuleOp moduleOp,
                                mlir::DataFlowSolver &solver,
                                mlir::func::FuncOp targetFunc) {
  mlir::MLIRContext *context = moduleOp.getContext();
  mlir::Location loc = targetFunc.getLoc();
  mlir::Type ptrTy = getOpaquePtrType(context);
  mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
  mlir::Type anyTy = libtriton::tvm_ffi::AnyType::get(context);
  mlir::Type anyLLVMType =
      libtriton::conversion::utils::TVMFFIAnyLLVMDescriptor::getLLVMType(
          context);

  const std::string wrapperName =
      (kTVMFFIWrapperPrefix + targetFunc.getSymName()).str();
  if (mlir::SymbolTable::lookupSymbolIn(moduleOp, wrapperName)) {
    targetFunc.emitError() << "duplicate wrapper symbol " << wrapperName;
    return mlir::failure();
  }

  mlir::FunctionType targetType = targetFunc.getFunctionType();
  if (targetType.getNumResults() > 1) {
    targetFunc.emitError()
        << "tvm_ffi.emit_tvm_ffi_interface only supports at most one return "
           "value";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Type> wrapperInputs = {ptrTy, ptrTy, i32Ty, ptrTy};
  llvm::SmallVector<mlir::Type> wrapperResults = {i32Ty};
  mlir::FunctionType wrapperType =
      mlir::FunctionType::get(context, wrapperInputs, wrapperResults);

  mlir::OpBuilder moduleBuilder(context);
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  mlir::func::FuncOp wrapperFunc =
      mlir::func::FuncOp::create(loc, wrapperName, wrapperType);
  moduleBuilder.insert(wrapperFunc);

  mlir::Block *entryBlock = wrapperFunc.addEntryBlock();
  mlir::OpBuilder invokeBuilder = mlir::OpBuilder::atBlockEnd(entryBlock);
  llvm::SmallVector<mlir::Value> callArgs;
  callArgs.reserve(targetType.getNumInputs());
  mlir::Value packedArgsPtr = entryBlock->getArgument(1);
  mlir::Value packedResultPtr = entryBlock->getArgument(3);
  for (std::int32_t i = 0; i < targetType.getNumInputs(); ++i) {
    mlir::TypedValue<mlir::IntegerType> argIndex =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            emitI64Constant(invokeBuilder, loc, i));
    mlir::Value argSlotPtr = emitAnyBufferSlotPtr(
        invokeBuilder, loc, packedArgsPtr, argIndex, anyLLVMType);
    mlir::Value anyValueLLVM =
        mlir::LLVM::LoadOp::create(invokeBuilder, loc, anyLLVMType, argSlotPtr)
            .getResult();
    mlir::Value anyValue = materializeCast(invokeBuilder, anyTy,
                                           mlir::ValueRange{anyValueLLVM}, loc);
    if (!anyValue)
      return mlir::failure();
    mlir::FailureOr<mlir::Value> unpackedValue =
        emitUnboxAnyValue(invokeBuilder, loc, targetType.getInput(i), anyValue);
    if (mlir::failed(unpackedValue))
      return mlir::failure();
    callArgs.push_back(*unpackedValue);
  }

  mlir::func::CallOp callOp =
      mlir::func::CallOp::create(invokeBuilder, loc, targetFunc, callArgs);
  if (callOp.getNumResults() == 1) {
    mlir::FailureOr<mlir::Value> boxedResult =
        emitBoxAnyValue(invokeBuilder, loc, solver, callOp.getResult(0));
    if (mlir::failed(boxedResult))
      return mlir::failure();
    mlir::TypedValue<mlir::IntegerType> resultIndex =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            emitI64Constant(invokeBuilder, loc, 0));
    mlir::Value resultSlotPtr = emitAnyBufferSlotPtr(
        invokeBuilder, loc, packedResultPtr, resultIndex, anyLLVMType);
    mlir::Value boxedResultLLVM = materializeCast(
        invokeBuilder, anyLLVMType, mlir::ValueRange{*boxedResult}, loc);
    if (!boxedResultLLVM)
      return mlir::failure();
    mlir::LLVM::StoreOp::create(invokeBuilder, loc, boxedResultLLVM,
                                resultSlotPtr);
  }
  mlir::Value successCode = emitI32Constant(invokeBuilder, loc, 0);
  mlir::func::ReturnOp::create(invokeBuilder, loc, successCode);
  return wrapperFunc;
}

class EmitTVMFFIInterfacePass
    : public impl::EmitTVMFFIInterfaceBase<EmitTVMFFIInterfacePass> {
public:
  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::DataFlowSolver solver;
    mlir::dataflow::loadBaselineAnalyses(solver);
    solver.load<libtriton::analysis::MemRefOriginDataFlowAnalysis>();
    if (mlir::failed(solver.initializeAndRun(moduleOp))) {
      signalPassFailure();
      return;
    }

    for (mlir::func::FuncOp targetFunc :
         llvm::make_filter_range(moduleOp.getOps<mlir::func::FuncOp>(),
                                 [](mlir::func::FuncOp funcOp) {
                                   return !funcOp.isDeclaration() &&
                                          funcOp->hasAttr(kTVMFFIInterfaceAttr);
                                 })) {
      if (mlir::failed(
              buildEmitTVMFFIInterfaceWrapper(moduleOp, solver, targetFunc))) {
        signalPassFailure();
        return;
      }
      targetFunc->removeAttr(kTVMFFIInterfaceAttr);
    }
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
  patterns
      .add<LowerFromFloatOp, LowerFromIntOp, LowerFromObjectOp, LowerFromStrOp,
           LowerFromTensorOp, LowerTensorFromDLPackOp, LowerToFloatOp,
           LowerToIntOp, LowerToObjectOp, LowerToStrOp, LowerToTensorOp>(
          typeConverter, context);

  target.addIllegalDialect<libtriton::tvm_ffi::TVMFFIDialect>();
}

void registerTVMFFIToLLVMPasses() {
  registerEmitTVMFFIInterfacePass();
  registerConvertTVMFFIToLLVMPass();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, libtriton::tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace libtriton::tvm_ffi
