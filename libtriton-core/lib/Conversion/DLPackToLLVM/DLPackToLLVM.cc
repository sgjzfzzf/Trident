// DLPackToLLVM.cc - Pass that lowers DLPack dialect ops to LLVM dialect.

#include <algorithm>
#include <cstdint>
#include <tuple>

#include "libtriton-core/Conversion/DLPackToLLVM/DLPackLLVMDescriptors.h"
#include "libtriton-core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "libtriton-core/Conversion/Utils/RuntimeCFunctionDeclUtils.h"
#include "libtriton-core/Conversion/Utils/StdLibCFunctionDeclUtils.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackDialect.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackOps.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::dlpack {

#define GEN_PASS_DEF_CONVERTDLPACKTOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
allocateArrayWithMalloc(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                        mlir::Location loc, const uint32_t count,
                        mlir::Type i64Ty, mlir::Type i8Ty, mlir::Type ptrTy) {
  // Call malloc(count * sizeof(int64_t))
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> mallocOrErr =
      conversion::utils::getOrCreateMalloc(moduleOp);
  if (mlir::failed(mallocOrErr)) {
    return nullptr;
  }

  mlir::Value sizeVal = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty,
                                                       count * sizeof(int64_t));
  mlir::LLVM::CallOp callOp = mlir::LLVM::CallOp::create(
      builder, loc, *mallocOrErr, mlir::ValueRange{sizeVal});

  return mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
      callOp.getResult());
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyShapeFromMemRefDescriptorToArray(mlir::OpBuilder &builder,
                                     mlir::ModuleOp moduleOp,
                                     mlir::Location loc,
                                     mlir::MemRefDescriptor memRefDescriptor,
                                     const uint32_t rank, mlir::Type i64Ty,
                                     mlir::Type i8Ty, mlir::Type ptrTy) {
  mlir::MLIRContext *context = moduleOp.getContext();

  mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeAlloc =
      allocateArrayWithMalloc(builder, moduleOp, loc, rank, i64Ty, i8Ty, ptrTy);
  if (!shapeAlloc) {
    return nullptr;
  }

  for (uint32_t i = 0; i < rank; ++i) {
    mlir::Value shapeElem = memRefDescriptor.size(builder, loc, i);

    mlir::Value shapeGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, shapeAlloc,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::LLVM::StoreOp::create(builder, loc, shapeElem, shapeGep);
  }

  return shapeAlloc;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyStrideFromMemRefDescriptorToArray(mlir::OpBuilder &builder,
                                      mlir::ModuleOp moduleOp,
                                      mlir::Location loc,
                                      mlir::MemRefDescriptor memRefDescriptor,
                                      const uint32_t rank, mlir::Type i64Ty,
                                      mlir::Type i8Ty, mlir::Type ptrTy) {
  mlir::MLIRContext *context = moduleOp.getContext();

  mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesAlloc =
      allocateArrayWithMalloc(builder, moduleOp, loc, rank, i64Ty, i8Ty, ptrTy);
  if (!stridesAlloc) {
    return nullptr;
  }

  for (uint32_t i = 0; i < rank; ++i) {
    mlir::Value strideElem = memRefDescriptor.stride(builder, loc, i);

    mlir::Value stridesGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, stridesAlloc,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::LLVM::StoreOp::create(builder, loc, strideElem, stridesGep);
  }

  return stridesAlloc;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyShapeFromMemRefDescriptorToStackArray(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::MemRefDescriptor memRefDescriptor, const uint32_t rank,
    mlir::Type i64Ty, mlir::Type ptrTy) {
  mlir::Value rankValue =
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, rank).getResult();
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeAlloca =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, i64Ty, rankValue)
              .getResult());
  for (uint32_t i = 0; i < rank; ++i) {
    mlir::Value shapeElem = memRefDescriptor.size(builder, loc, i);
    mlir::Value shapeGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, shapeAlloca,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::LLVM::StoreOp::create(builder, loc, shapeElem, shapeGep);
  }
  return shapeAlloca;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyStrideFromMemRefDescriptorToStackArray(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::MemRefDescriptor memRefDescriptor, const uint32_t rank,
    mlir::Type i64Ty, mlir::Type ptrTy) {
  mlir::Value rankValue =
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, rank).getResult();
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesAlloca =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, i64Ty, rankValue)
              .getResult());
  for (uint32_t i = 0; i < rank; ++i) {
    mlir::Value strideElem = memRefDescriptor.stride(builder, loc, i);
    mlir::Value stridesGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, stridesAlloca,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::LLVM::StoreOp::create(builder, loc, strideElem, stridesGep);
  }
  return stridesAlloca;
}

std::tuple<llvm::SmallVector<mlir::Value>, llvm::SmallVector<mlir::Value>>
extractShapeAndStrideFromArrays(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapePtr,
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesPtr,
    const uint32_t rank, mlir::Type i64Ty, mlir::Type ptrTy) {
  llvm::SmallVector<mlir::Value> shapeValues;
  llvm::SmallVector<mlir::Value> strideValues;

  for (uint32_t i = 0; i < rank; ++i) {
    mlir::Value shapeGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, shapePtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    shapeValues.emplace_back(
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, shapeGep));
  }

  mlir::Value rankValue =
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, rank).getResult();
  mlir::Value fallbackStridesAlloca =
      mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, i64Ty, rankValue)
          .getResult();

  mlir::Value runningStride =
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1).getResult();
  for (int32_t i = rank - 1; i >= 0; --i) {
    mlir::Value strideGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, fallbackStridesAlloca,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
    mlir::LLVM::StoreOp::create(builder, loc, runningStride, strideGep);
    runningStride = mlir::LLVM::MulOp::create(builder, loc, i64Ty,
                                              runningStride, shapeValues[i]);
  }

  mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(builder, loc, ptrTy);
  mlir::Value useFallback = mlir::LLVM::ICmpOp::create(
      builder, loc, mlir::LLVM::ICmpPredicate::eq, stridesPtr, nullPtr);
  mlir::Value effectiveStridesPtr = mlir::LLVM::SelectOp::create(
      builder, loc, ptrTy, useFallback, fallbackStridesAlloca, stridesPtr);

  for (uint32_t i = 0; i < rank; ++i) {
    mlir::Value stridesGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, effectiveStridesPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    strideValues.emplace_back(
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, stridesGep));
  }

  return std::make_tuple(shapeValues, strideValues);
}

struct LowerFromMemRefOwnedOp
    : public mlir::OpConversionPattern<FromMemRefOwnedOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FromMemRefOwnedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
      return mlir::failure();

    mlir::MemRefType memRefType =
        mlir::cast<mlir::MemRefType>(op.getInput().getType());
    const uint32_t rank = static_cast<uint32_t>(memRefType.getRank());

    // adaptor.getInput() is the LLVM memref descriptor struct after type
    // conversion
    mlir::MemRefDescriptor memDesc(adaptor.getInput());

    mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
    mlir::Type i16Ty = mlir::IntegerType::get(context, 16);
    mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    const mlir::LLVMTypeConverter *llvmTypeConverter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());

    mlir::Type convertedManagedTensorType =
        llvmTypeConverter->convertType(op.getOutput().getType());
    mlir::Type convertedDLTensorType =
        conversion::utils::DLTensorLLVMDescriptor::getLLVMType(context);
    mlir::Type convertedDLDeviceType =
        conversion::utils::DLDeviceLLVMDescriptor::getLLVMType(context);
    mlir::Type convertedDLDataTypeType =
        conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(context);

    mlir::LLVM::LLVMStructType dlManagedTensorTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedManagedTensorType);
    mlir::LLVM::LLVMStructType dlTensorTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLTensorType);
    mlir::LLVM::LLVMStructType dlDeviceTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDeviceType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlManagedTensorTy || !dlTensorTy || !dlDeviceTy || !dlDataTypeTy)
      return mlir::failure();

    // Determine DLDataType fields from the memref element type
    uint8_t dtypeCode;
    uint8_t dtypeBits;
    const uint16_t dtypeLanes = 1;
    mlir::Type elemTy = memRefType.getElementType();
    if (elemTy.isF16()) {
      dtypeCode = kDLFloat;
      dtypeBits = 16;
    } else if (elemTy.isF32()) {
      dtypeCode = kDLFloat;
      dtypeBits = 32;
    } else if (elemTy.isF64()) {
      dtypeCode = kDLFloat;
      dtypeBits = 64;
    } else if (mlir::IntegerType integerType =
                   mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      dtypeCode = integerType.isUnsigned() ? kDLUInt : kDLInt;
      dtypeBits = static_cast<uint8_t>(integerType.getWidth());
    } else {
      return mlir::failure();
    }

    // Extract aligned pointer and element offset from the memref descriptor
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> dataPtr =
        mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
            memDesc.alignedPtr(rewriter, loc));
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> allocatedPtr =
        mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
            memDesc.allocatedPtr(rewriter, loc));
    mlir::TypedValue<mlir::IntegerType> elemOffset =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            memDesc.offset(rewriter, loc));

    const int64_t elemByteWidth = std::max<int64_t>(1, (dtypeBits + 7) / 8);
    mlir::TypedValue<mlir::IntegerType> elemByteWidthValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, elemByteWidth)
                .getResult());

    mlir::TypedValue<mlir::IntegerType> byteOffsetValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::MulOp::create(rewriter, loc, elemOffset,
                                      elemByteWidthValue)
                .getResult());

    mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeAlloca =
        copyShapeFromMemRefDescriptorToArray(rewriter, moduleOp, loc, memDesc,
                                             rank, i64Ty, i8Ty, ptrTy);
    if (!shapeAlloca)
      return mlir::failure();

    mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesAlloca =
        copyStrideFromMemRefDescriptorToArray(rewriter, moduleOp, loc, memDesc,
                                              rank, i64Ty, i8Ty, ptrTy);
    if (!stridesAlloca)
      return mlir::failure();

    // Build DLDevice struct: {device_type = kDLCUDA, device_id = 0}
    mlir::TypedValue<mlir::IntegerType> deviceTypeValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kDLCUDA)
                .getResult());
    mlir::TypedValue<mlir::IntegerType> deviceIdValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0LL)
                .getResult());
    conversion::utils::DLDeviceLLVMDescriptor dlDevice =
        conversion::utils::DLDeviceLLVMDescriptor::build(
            rewriter, loc, dlDeviceTy, deviceTypeValue, deviceIdValue);

    // Build DLDataType struct: {code, bits, lanes = 1}
    mlir::TypedValue<mlir::IntegerType> dtypeCodeValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty, dtypeCode)
                .getResult());
    mlir::TypedValue<mlir::IntegerType> dtypeBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty, dtypeBits)
                .getResult());
    mlir::TypedValue<mlir::IntegerType> dtypeLanesValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i16Ty, dtypeLanes)
                .getResult());
    conversion::utils::DLDataTypeLLVMDescriptor dlDataType =
        conversion::utils::DLDataTypeLLVMDescriptor::build(
            rewriter, loc, dlDataTypeTy, dtypeCodeValue, dtypeBitsValue,
            dtypeLanesValue);

    mlir::TypedValue<mlir::IntegerType> ndimValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, rank)
                .getResult());

    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::build(
            rewriter, loc, dlTensorTy, dataPtr, dlDevice, ndimValue, dlDataType,
            shapeAlloca, stridesAlloca, byteOffsetValue);

    // Preserve the original allocation base pointer so runtime deleters can
    // free the correct address even when tensor data points to an aligned view.
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> managerCtxValue =
        allocatedPtr;
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> deleterOrErr =
        conversion::utils::runtime::getOrCreateDefaultManagedTensorDeleter(
            moduleOp);
    if (mlir::failed(deleterOrErr))
      return mlir::failure();

    mlir::TypedValue<mlir::LLVM::LLVMPointerType> deleterValue =
        mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
            mlir::LLVM::AddressOfOp::create(rewriter, loc, *deleterOrErr)
                .getResult());

    conversion::utils::DLManagedTensorLLVMDescriptor dlManagedTensor =
        conversion::utils::DLManagedTensorLLVMDescriptor::build(
            rewriter, loc, dlManagedTensorTy, dlTensor, managerCtxValue,
            deleterValue);

    rewriter.replaceOp(op, dlManagedTensor.as());
    return mlir::success();
  }
};

struct LowerViewOp : public mlir::OpConversionPattern<ViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLManagedTensorLLVMDescriptor dlManagedTensor =
        conversion::utils::DLManagedTensorLLVMDescriptor::from(
            adaptor.getInput());
    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        dlManagedTensor.tensor(rewriter, loc);
    rewriter.replaceOp(op, dlTensor.as());
    return mlir::success();
  }
};

struct LowerFromMemRefBorrowedOp
    : public mlir::OpConversionPattern<FromMemRefBorrowedOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FromMemRefBorrowedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    mlir::MemRefType memRefType =
        mlir::cast<mlir::MemRefType>(op.getInput().getType());
    const uint32_t rank = static_cast<uint32_t>(memRefType.getRank());

    mlir::MemRefDescriptor memDesc(adaptor.getInput());

    mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
    mlir::Type i16Ty = mlir::IntegerType::get(context, 16);
    mlir::Type i32Ty = mlir::IntegerType::get(context, 32);
    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    const mlir::LLVMTypeConverter *llvmTypeConverter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    mlir::Type convertedDLTensorType =
        llvmTypeConverter->convertType(op.getOutput().getType());
    mlir::Type convertedDLDeviceType =
        conversion::utils::DLDeviceLLVMDescriptor::getLLVMType(context);
    mlir::Type convertedDLDataTypeType =
        conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(context);

    mlir::LLVM::LLVMStructType dlTensorTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLTensorType);
    mlir::LLVM::LLVMStructType dlDeviceTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDeviceType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlTensorTy || !dlDeviceTy || !dlDataTypeTy)
      return mlir::failure();

    uint8_t dtypeCode;
    uint8_t dtypeBits;
    const uint16_t dtypeLanes = 1;
    mlir::Type elemTy = memRefType.getElementType();
    if (elemTy.isF16()) {
      dtypeCode = kDLFloat;
      dtypeBits = 16;
    } else if (elemTy.isF32()) {
      dtypeCode = kDLFloat;
      dtypeBits = 32;
    } else if (elemTy.isF64()) {
      dtypeCode = kDLFloat;
      dtypeBits = 64;
    } else if (mlir::IntegerType integerType =
                   mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      dtypeCode = integerType.isUnsigned() ? kDLUInt : kDLInt;
      dtypeBits = static_cast<uint8_t>(integerType.getWidth());
    } else {
      return mlir::failure();
    }

    mlir::TypedValue<mlir::LLVM::LLVMPointerType> dataPtr =
        mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
            memDesc.alignedPtr(rewriter, loc));
    mlir::TypedValue<mlir::IntegerType> elemOffset =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            memDesc.offset(rewriter, loc));

    const int64_t elemByteWidth = std::max<int64_t>(1, (dtypeBits + 7) / 8);
    mlir::TypedValue<mlir::IntegerType> elemByteWidthValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, elemByteWidth)
                .getResult());
    mlir::TypedValue<mlir::IntegerType> byteOffsetValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::MulOp::create(rewriter, loc, elemOffset,
                                      elemByteWidthValue)
                .getResult());

    mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeAlloca =
        copyShapeFromMemRefDescriptorToStackArray(rewriter, loc, memDesc, rank,
                                                  i64Ty, ptrTy);
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesAlloca =
        copyStrideFromMemRefDescriptorToStackArray(rewriter, loc, memDesc, rank,
                                                   i64Ty, ptrTy);

    mlir::TypedValue<mlir::IntegerType> deviceTypeValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, kDLCPU)
                .getResult());
    mlir::TypedValue<mlir::IntegerType> deviceIdValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0LL)
                .getResult());
    conversion::utils::DLDeviceLLVMDescriptor dlDevice =
        conversion::utils::DLDeviceLLVMDescriptor::build(
            rewriter, loc, dlDeviceTy, deviceTypeValue, deviceIdValue);

    mlir::TypedValue<mlir::IntegerType> dtypeCodeValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty, dtypeCode)
                .getResult());
    mlir::TypedValue<mlir::IntegerType> dtypeBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty, dtypeBits)
                .getResult());
    mlir::TypedValue<mlir::IntegerType> dtypeLanesValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i16Ty, dtypeLanes)
                .getResult());
    conversion::utils::DLDataTypeLLVMDescriptor dlDataType =
        conversion::utils::DLDataTypeLLVMDescriptor::build(
            rewriter, loc, dlDataTypeTy, dtypeCodeValue, dtypeBitsValue,
            dtypeLanesValue);

    mlir::TypedValue<mlir::IntegerType> ndimValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, rank)
                .getResult());

    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::build(
            rewriter, loc, dlTensorTy, dataPtr, dlDevice, ndimValue, dlDataType,
            shapeAlloca, stridesAlloca, byteOffsetValue);

    rewriter.replaceOp(op, dlTensor.as());
    return mlir::success();
  }
};

struct LowerToMemRefOp : public mlir::OpConversionPattern<ToMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ToMemRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    mlir::MemRefType memRefType =
        mlir::cast<mlir::MemRefType>(op.getOutput().getType());
    const uint32_t rank = static_cast<uint32_t>(memRefType.getRank());

    mlir::Type outputType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!outputType) {
      return mlir::failure();
    }

    mlir::Type convertedDLDeviceType =
        conversion::utils::DLDeviceLLVMDescriptor::getLLVMType(context);
    mlir::Type convertedDLDataTypeType =
        conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(context);
    mlir::LLVM::LLVMStructType dlDeviceTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDeviceType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlDeviceTy || !dlDataTypeTy) {
      return mlir::failure();
    }

    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::from(adaptor.getInput());
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> dataPtr =
        dlTensor.data(rewriter, loc);
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapePtr =
        dlTensor.shape(rewriter, loc);
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesPtr =
        dlTensor.strides(rewriter, loc);
    mlir::TypedValue<mlir::IntegerType> byteOffset =
        dlTensor.byteOffset(rewriter, loc);

    // Build an LLVM memref descriptor from the DLTensor fields
    mlir::MemRefDescriptor memDesc =
        mlir::MemRefDescriptor::poison(rewriter, loc, outputType);
    memDesc.setAllocatedPtr(rewriter, loc, dataPtr);
    memDesc.setAlignedPtr(rewriter, loc, dataPtr);
    memDesc.setOffset(rewriter, loc, byteOffset);

    std::tuple<llvm::SmallVector<mlir::Value>, llvm::SmallVector<mlir::Value>>
        shapeAndStrideValues = extractShapeAndStrideFromArrays(
            rewriter, loc, shapePtr, stridesPtr, rank, i64Ty, ptrTy);
    auto [shapeValues, strideValues] = std::move(shapeAndStrideValues);
    for (uint32_t i = 0; i < rank; ++i) {
      memDesc.setSize(rewriter, loc, i, shapeValues[i]);
      memDesc.setStride(rewriter, loc, i, strideValues[i]);
    }

    rewriter.replaceOp(op, mlir::Value(memDesc));
    return mlir::success();
  }
};

struct LowerNDimOp : public mlir::OpConversionPattern<NDimOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NDimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlTensor.ndim(rewriter, loc));
    return mlir::success();
  }
};

struct LowerShapeOp : public mlir::OpConversionPattern<ShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ShapeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::from(adaptor.getInput());
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapePtr =
        dlTensor.shape(rewriter, loc);
    mlir::Value indexValue = adaptor.getIndex();
    mlir::Type i64Ty = mlir::IntegerType::get(op.getContext(), 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(op.getContext());
    mlir::Value elementPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i64Ty, shapePtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{indexValue});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, i64Ty, elementPtr);
    return mlir::success();
  }
};

struct LowerStridesOp : public mlir::OpConversionPattern<StridesOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(StridesOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::from(adaptor.getInput());
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesPtr =
        dlTensor.strides(rewriter, loc);
    mlir::Value indexValue = adaptor.getIndex();
    mlir::Type i64Ty = mlir::IntegerType::get(op.getContext(), 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(op.getContext());
    mlir::Value elementPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i64Ty, stridesPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{indexValue});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, i64Ty, elementPtr);
    return mlir::success();
  }
};

struct LowerByteOffsetOp : public mlir::OpConversionPattern<ByteOffsetOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ByteOffsetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlTensor.byteOffset(rewriter, loc));
    return mlir::success();
  }
};

struct LowerDTypeOp : public mlir::OpConversionPattern<DTypeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DTypeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlTensor.dtype(rewriter, loc).as());
    return mlir::success();
  }
};

struct LowerDTypeCodeOp : public mlir::OpConversionPattern<DTypeCodeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DTypeCodeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLDataTypeLLVMDescriptor dlDataType =
        conversion::utils::DLDataTypeLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlDataType.code(rewriter, loc));
    return mlir::success();
  }
};

struct LowerDTypeBitsOp : public mlir::OpConversionPattern<DTypeBitsOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DTypeBitsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLDataTypeLLVMDescriptor dlDataType =
        conversion::utils::DLDataTypeLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlDataType.bits(rewriter, loc));
    return mlir::success();
  }
};

struct LowerDTypeLanesOp : public mlir::OpConversionPattern<DTypeLanesOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DTypeLanesOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLDataTypeLLVMDescriptor dlDataType =
        conversion::utils::DLDataTypeLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlDataType.lanes(rewriter, loc));
    return mlir::success();
  }
};

struct LowerDeviceOp : public mlir::OpConversionPattern<DeviceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DeviceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLTensorLLVMDescriptor dlTensor =
        conversion::utils::DLTensorLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlTensor.device(rewriter, loc).as());
    return mlir::success();
  }
};

struct LowerDeviceTypeOp : public mlir::OpConversionPattern<DeviceTypeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DeviceTypeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLDeviceLLVMDescriptor dlDevice =
        conversion::utils::DLDeviceLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlDevice.deviceType(rewriter, loc));
    return mlir::success();
  }
};

struct LowerDeviceIdOp : public mlir::OpConversionPattern<DeviceIdOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DeviceIdOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    conversion::utils::DLDeviceLLVMDescriptor dlDevice =
        conversion::utils::DLDeviceLLVMDescriptor::from(adaptor.getInput());
    rewriter.replaceOp(op, dlDevice.deviceId(rewriter, loc));
    return mlir::success();
  }
};

struct LowerTensorFromLLVMOp
    : public mlir::OpConversionPattern<TensorFromLLVMOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorFromLLVMOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type convertedTensorType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedTensorType ||
        adaptor.getInput().getType() != convertedTensorType) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

struct LowerTensorToLLVMOp : public mlir::OpConversionPattern<TensorToLLVMOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorToLLVMOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getInput().getType() != op.getOutput().getType()) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

class ConvertDLPackToLLVMPass
    : public impl::ConvertDLPackToLLVMBase<ConvertDLPackToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    populateDLPackToLLVMConversionPatterns(target, typeConverter, patterns);

    if (failed(mlir::applyPartialConversion(getOperation(), target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct DLPackToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateDLPackToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateDLPackToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  typeConverter.addConversion([](DLDeviceType type) {
    return conversion::utils::DLDeviceLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  typeConverter.addConversion([](DLDataTypeType type) {
    return conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  typeConverter.addConversion([&](DLTensorType type) -> mlir::Type {
    return conversion::utils::DLTensorLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  typeConverter.addConversion([&](DLManagedTensorType type) -> mlir::Type {
    return conversion::utils::DLManagedTensorLLVMDescriptor::getLLVMType(
        type.getContext());
  });

  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, typeConverter);
  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<LowerFromMemRefOwnedOp, LowerFromMemRefBorrowedOp,
               LowerTensorFromLLVMOp, LowerTensorToLLVMOp, LowerViewOp,
               LowerToMemRefOp, LowerNDimOp, LowerShapeOp, LowerStridesOp,
               LowerByteOffsetOp, LowerDTypeOp, LowerDTypeCodeOp,
               LowerDTypeBitsOp, LowerDTypeLanesOp, LowerDeviceOp,
               LowerDeviceTypeOp, LowerDeviceIdOp>(typeConverter, context);

  target.addIllegalDialect<DLPackDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
  target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<mlir::func::CallOp>([&](mlir::func::CallOp op) {
    return typeConverter.isLegal(op.getOperandTypes()) &&
           typeConverter.isLegal(op.getResultTypes());
  });
  target.addDynamicallyLegalOp<mlir::func::ReturnOp>([&](mlir::Operation *op) {
    return mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter);
  });
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

void registerConvertDLPackToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, DLPackDialect *dialect) {
    dialect->addInterfaces<DLPackToLLVMDialectInterface>();
  });
}

} // namespace libtriton::dlpack
