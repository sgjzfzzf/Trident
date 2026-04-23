// DLPackToLLVM.cc - Pass that lowers DLPack dialect ops to LLVM dialect.

#include <algorithm>
#include <cstdint>
#include <tuple>

#include "libtriton-core/Conversion/DLPackToLLVM/DLPackLLVMDescriptors.h"
#include "libtriton-core/Conversion/DLPackToLLVM/DLPackToLLVM.h"
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

constexpr const char kDefaultManagedTensorDeleterName[] =
    "__libtriton_dlpack_default_managed_tensor_deleter";

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
allocateArrayWithMalloc(mlir::ConversionPatternRewriter &rewriter,
                        mlir::ModuleOp moduleOp, mlir::Location loc,
                        const std::uint32_t count, mlir::Type i64Ty,
                        mlir::Type i8Ty, mlir::Type ptrTy) {
  // Call malloc(count * sizeof(int64_t))
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> mallocOrErr =
      libtriton::conversion::utils::getOrCreateMalloc(moduleOp);
  if (mlir::failed(mallocOrErr))
    return nullptr;

  mlir::Value sizeVal = mlir::LLVM::ConstantOp::create(
      rewriter, loc, i64Ty, static_cast<int64_t>(count * sizeof(int64_t)));
  mlir::LLVM::CallOp callOp = mlir::LLVM::CallOp::create(
      rewriter, loc, *mallocOrErr, mlir::ValueRange{sizeVal});

  return mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
      callOp.getResult());
}

mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getOrCreateDefaultManagedTensorDeleter(
    mlir::ModuleOp moduleOp, mlir::LLVM::LLVMStructType dlManagedTensorTy) {
  mlir::MLIRContext *context = moduleOp.getContext();
  mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(context);
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  mlir::LLVM::LLVMFunctionType funcType = mlir::LLVM::LLVMFunctionType::get(
      voidTy, llvm::SmallVector<mlir::Type>{ptrTy});

  mlir::LLVM::LLVMFuncOp existingFunc =
      moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          kDefaultManagedTensorDeleterName);
  if (existingFunc) {
    if (existingFunc.getFunctionType() != funcType) {
      moduleOp.emitError() << "existing llvm.func @"
                           << kDefaultManagedTensorDeleterName
                           << " has incompatible signature";
      return mlir::failure();
    }
    return existingFunc;
  }

  mlir::OpBuilder builder(context);
  builder.setInsertionPointToStart(moduleOp.getBody());
  mlir::LLVM::LLVMFuncOp deleterDecl = mlir::LLVM::LLVMFuncOp::create(
      builder, moduleOp.getLoc(), kDefaultManagedTensorDeleterName, funcType,
      mlir::LLVM::Linkage::External);
  return deleterDecl;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyShapeFromMemRefDescriptorToArray(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::ModuleOp moduleOp,
                                     mlir::Location loc,
                                     mlir::MemRefDescriptor memRefDescriptor,
                                     const std::uint32_t rank, mlir::Type i64Ty,
                                     mlir::Type i8Ty, mlir::Type ptrTy) {
  mlir::MLIRContext *context = moduleOp.getContext();

  mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeAlloc =
      allocateArrayWithMalloc(rewriter, moduleOp, loc, rank, i64Ty, i8Ty,
                              ptrTy);
  if (!shapeAlloc)
    return nullptr;

  for (std::uint32_t i = 0; i < rank; ++i) {
    mlir::Value shapeElem = memRefDescriptor.size(rewriter, loc, i);

    mlir::Value shapeGep = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i64Ty, shapeAlloc,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::LLVM::StoreOp::create(rewriter, loc, shapeElem, shapeGep);
  }

  return shapeAlloc;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyStrideFromMemRefDescriptorToArray(mlir::ConversionPatternRewriter &rewriter,
                                      mlir::ModuleOp moduleOp,
                                      mlir::Location loc,
                                      mlir::MemRefDescriptor memRefDescriptor,
                                      const std::uint32_t rank,
                                      mlir::Type i64Ty, mlir::Type i8Ty,
                                      mlir::Type ptrTy) {
  mlir::MLIRContext *context = moduleOp.getContext();

  mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesAlloc =
      allocateArrayWithMalloc(rewriter, moduleOp, loc, rank, i64Ty, i8Ty,
                              ptrTy);
  if (!stridesAlloc)
    return nullptr;

  for (std::uint32_t i = 0; i < rank; ++i) {
    mlir::Value strideElem = memRefDescriptor.stride(rewriter, loc, i);

    mlir::Value stridesGep = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i64Ty, stridesAlloc,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::LLVM::StoreOp::create(rewriter, loc, strideElem, stridesGep);
  }

  return stridesAlloc;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyShapeFromMemRefDescriptorToStackArray(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::MemRefDescriptor memRefDescriptor, const std::uint32_t rank,
    mlir::Type i64Ty, mlir::Type ptrTy) {
  mlir::Value rankValue =
      mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, rank).getResult();
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeAlloca =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, rankValue)
              .getResult());
  for (std::uint32_t i = 0; i < rank; ++i) {
    mlir::Value shapeElem = memRefDescriptor.size(rewriter, loc, i);
    mlir::Value shapeGep = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i64Ty, shapeAlloca,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::LLVM::StoreOp::create(rewriter, loc, shapeElem, shapeGep);
  }
  return shapeAlloca;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyStrideFromMemRefDescriptorToStackArray(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::MemRefDescriptor memRefDescriptor, const std::uint32_t rank,
    mlir::Type i64Ty, mlir::Type ptrTy) {
  mlir::Value rankValue =
      mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, rank).getResult();
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesAlloca =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, rankValue)
              .getResult());
  for (std::uint32_t i = 0; i < rank; ++i) {
    mlir::Value strideElem = memRefDescriptor.stride(rewriter, loc, i);
    mlir::Value stridesGep = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i64Ty, stridesAlloca,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::LLVM::StoreOp::create(rewriter, loc, strideElem, stridesGep);
  }
  return stridesAlloca;
}

std::tuple<llvm::SmallVector<mlir::Value>, llvm::SmallVector<mlir::Value>>
extractShapeAndStrideFromArrays(
    mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapePtr,
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesPtr,
    const std::uint32_t rank, mlir::Type i64Ty, mlir::Type ptrTy) {
  llvm::SmallVector<mlir::Value> shapeValues;
  llvm::SmallVector<mlir::Value> strideValues;
  shapeValues.reserve(rank);
  strideValues.reserve(rank);

  for (std::uint32_t i = 0; i < rank; ++i) {
    mlir::Value shapeGep = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i64Ty, shapePtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::Value shapeElem =
        mlir::LLVM::LoadOp::create(rewriter, loc, i64Ty, shapeGep);
    shapeValues.push_back(shapeElem);

    mlir::Value stridesGep = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, i64Ty, stridesPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::Value strideElem =
        mlir::LLVM::LoadOp::create(rewriter, loc, i64Ty, stridesGep);
    strideValues.push_back(strideElem);
  }

  return std::make_tuple(shapeValues, strideValues);
}

struct LowerFromMemRefOwnedOp
    : public mlir::OpConversionPattern<libtriton::dlpack::FromMemRefOwnedOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::FromMemRefOwnedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();
    mlir::ModuleOp moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
      return mlir::failure();

    mlir::MemRefType memRefType =
        mlir::cast<mlir::MemRefType>(op.getInput().getType());
    const std::uint32_t rank = static_cast<std::uint32_t>(memRefType.getRank());

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
        libtriton::conversion::utils::DLTensorLLVMDescriptor::getLLVMType(
            context, llvmTypeConverter->getPointerBitwidth());
    mlir::Type convertedDLContextType =
        libtriton::conversion::utils::DLContextLLVMDescriptor::getLLVMType(
            context);
    mlir::Type convertedDLDataTypeType =
        libtriton::conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(
            context);

    mlir::LLVM::LLVMStructType dlManagedTensorTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedManagedTensorType);
    mlir::LLVM::LLVMStructType dlTensorTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLTensorType);
    mlir::LLVM::LLVMStructType dlContextTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLContextType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlManagedTensorTy || !dlTensorTy || !dlContextTy || !dlDataTypeTy)
      return mlir::failure();

    // Determine DLDataType fields from the memref element type
    std::uint8_t dtypeCode;
    std::uint8_t dtypeBits;
    const std::uint16_t dtypeLanes = 1;
    mlir::Type elemTy = memRefType.getElementType();
    if (elemTy.isF16()) {
      dtypeCode = static_cast<std::uint8_t>(kDLFloat);
      dtypeBits = 16;
    } else if (elemTy.isF32()) {
      dtypeCode = static_cast<std::uint8_t>(kDLFloat);
      dtypeBits = 32;
    } else if (elemTy.isF64()) {
      dtypeCode = static_cast<std::uint8_t>(kDLFloat);
      dtypeBits = 64;
    } else if (mlir::IntegerType integerType =
                   mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      dtypeCode = integerType.isUnsigned() ? static_cast<std::uint8_t>(kDLUInt)
                                           : static_cast<std::uint8_t>(kDLInt);
      dtypeBits = static_cast<std::uint8_t>(integerType.getWidth());
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

    const std::uint64_t elemByteWidth = std::max<std::uint64_t>(
        1, (static_cast<std::uint64_t>(dtypeBits) + 7) / 8);
    mlir::TypedValue<mlir::IntegerType> elemByteWidthValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           static_cast<int64_t>(elemByteWidth))
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

    // Build DLContext struct: {device_type = kCPU = 1, device_id = 0}
    mlir::TypedValue<mlir::IntegerType> deviceTypeValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           static_cast<std::int32_t>(kDLCPU))
                .getResult());
    mlir::TypedValue<mlir::IntegerType> deviceIdValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0LL)
                .getResult());
    libtriton::conversion::utils::DLContextLLVMDescriptor dlContext =
        libtriton::conversion::utils::DLContextLLVMDescriptor::build(
            rewriter, loc, dlContextTy, deviceTypeValue, deviceIdValue);

    // Build DLDataType struct: {code, bits, lanes = 1}
    mlir::TypedValue<mlir::IntegerType> dtypeCodeValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                           static_cast<int64_t>(dtypeCode))
                .getResult());
    mlir::TypedValue<mlir::IntegerType> dtypeBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                           static_cast<int64_t>(dtypeBits))
                .getResult());
    mlir::TypedValue<mlir::IntegerType> dtypeLanesValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i16Ty,
                                           static_cast<int64_t>(dtypeLanes))
                .getResult());
    libtriton::conversion::utils::DLDataTypeLLVMDescriptor dlDataType =
        libtriton::conversion::utils::DLDataTypeLLVMDescriptor::build(
            rewriter, loc, dlDataTypeTy, dtypeCodeValue, dtypeBitsValue,
            dtypeLanesValue);

    mlir::TypedValue<mlir::IntegerType> ndimValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           static_cast<int64_t>(rank))
                .getResult());

    libtriton::conversion::utils::DLTensorLLVMDescriptor dlTensor =
        libtriton::conversion::utils::DLTensorLLVMDescriptor::build(
            rewriter, loc, dlTensorTy, dataPtr, dlContext, ndimValue,
            dlDataType, shapeAlloca, stridesAlloca, byteOffsetValue);

    // Preserve the original allocation base pointer so runtime deleters can
    // free the correct address even when tensor data points to an aligned view.
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> managerCtxValue =
        allocatedPtr;
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> deleterOrErr =
        getOrCreateDefaultManagedTensorDeleter(moduleOp, dlManagedTensorTy);
    if (mlir::failed(deleterOrErr))
      return mlir::failure();

    mlir::TypedValue<mlir::LLVM::LLVMPointerType> deleterValue =
        mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
            mlir::LLVM::AddressOfOp::create(rewriter, loc, *deleterOrErr)
                .getResult());

    libtriton::conversion::utils::DLManagedTensorLLVMDescriptor
        dlManagedTensor =
            libtriton::conversion::utils::DLManagedTensorLLVMDescriptor::build(
                rewriter, loc, dlManagedTensorTy, dlTensor, managerCtxValue,
                deleterValue);

    rewriter.replaceOp(op, dlManagedTensor.as());
    return mlir::success();
  }
};

struct LowerViewOp
    : public mlir::OpConversionPattern<libtriton::dlpack::ViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::ViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    libtriton::conversion::utils::DLManagedTensorLLVMDescriptor
        dlManagedTensor =
            libtriton::conversion::utils::DLManagedTensorLLVMDescriptor::from(
                adaptor.getInput());
    libtriton::conversion::utils::DLTensorLLVMDescriptor dlTensor =
        dlManagedTensor.tensor(rewriter, loc);
    rewriter.replaceOp(op, dlTensor.as());
    return mlir::success();
  }
};

struct LowerFromMemRefBorrowedOp
    : public mlir::OpConversionPattern<
          libtriton::dlpack::FromMemRefBorrowedOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::FromMemRefBorrowedOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    mlir::MemRefType memRefType =
        mlir::cast<mlir::MemRefType>(op.getInput().getType());
    const std::uint32_t rank = static_cast<std::uint32_t>(memRefType.getRank());

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
    mlir::Type convertedDLContextType =
        libtriton::conversion::utils::DLContextLLVMDescriptor::getLLVMType(
            context);
    mlir::Type convertedDLDataTypeType =
        libtriton::conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(
            context);

    mlir::LLVM::LLVMStructType dlTensorTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLTensorType);
    mlir::LLVM::LLVMStructType dlContextTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLContextType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlTensorTy || !dlContextTy || !dlDataTypeTy)
      return mlir::failure();

    std::uint8_t dtypeCode;
    std::uint8_t dtypeBits;
    const std::uint16_t dtypeLanes = 1;
    mlir::Type elemTy = memRefType.getElementType();
    if (elemTy.isF16()) {
      dtypeCode = static_cast<std::uint8_t>(kDLFloat);
      dtypeBits = 16;
    } else if (elemTy.isF32()) {
      dtypeCode = static_cast<std::uint8_t>(kDLFloat);
      dtypeBits = 32;
    } else if (elemTy.isF64()) {
      dtypeCode = static_cast<std::uint8_t>(kDLFloat);
      dtypeBits = 64;
    } else if (mlir::IntegerType integerType =
                   mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      dtypeCode = integerType.isUnsigned() ? static_cast<std::uint8_t>(kDLUInt)
                                           : static_cast<std::uint8_t>(kDLInt);
      dtypeBits = static_cast<std::uint8_t>(integerType.getWidth());
    } else {
      return mlir::failure();
    }

    mlir::TypedValue<mlir::LLVM::LLVMPointerType> dataPtr =
        mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
            memDesc.alignedPtr(rewriter, loc));
    mlir::TypedValue<mlir::IntegerType> elemOffset =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            memDesc.offset(rewriter, loc));

    const std::uint64_t elemByteWidth = std::max<std::uint64_t>(
        1, (static_cast<std::uint64_t>(dtypeBits) + 7) / 8);
    mlir::TypedValue<mlir::IntegerType> elemByteWidthValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           static_cast<int64_t>(elemByteWidth))
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
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           static_cast<std::int32_t>(kDLCPU))
                .getResult());
    mlir::TypedValue<mlir::IntegerType> deviceIdValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0LL)
                .getResult());
    libtriton::conversion::utils::DLContextLLVMDescriptor dlContext =
        libtriton::conversion::utils::DLContextLLVMDescriptor::build(
            rewriter, loc, dlContextTy, deviceTypeValue, deviceIdValue);

    mlir::TypedValue<mlir::IntegerType> dtypeCodeValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                           static_cast<int64_t>(dtypeCode))
                .getResult());
    mlir::TypedValue<mlir::IntegerType> dtypeBitsValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                           static_cast<int64_t>(dtypeBits))
                .getResult());
    mlir::TypedValue<mlir::IntegerType> dtypeLanesValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i16Ty,
                                           static_cast<int64_t>(dtypeLanes))
                .getResult());
    libtriton::conversion::utils::DLDataTypeLLVMDescriptor dlDataType =
        libtriton::conversion::utils::DLDataTypeLLVMDescriptor::build(
            rewriter, loc, dlDataTypeTy, dtypeCodeValue, dtypeBitsValue,
            dtypeLanesValue);

    mlir::TypedValue<mlir::IntegerType> ndimValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           static_cast<int64_t>(rank))
                .getResult());

    libtriton::conversion::utils::DLTensorLLVMDescriptor dlTensor =
        libtriton::conversion::utils::DLTensorLLVMDescriptor::build(
            rewriter, loc, dlTensorTy, dataPtr, dlContext, ndimValue,
            dlDataType, shapeAlloca, stridesAlloca, byteOffsetValue);

    rewriter.replaceOp(op, dlTensor.as());
    return mlir::success();
  }
};

struct LowerToMemRefOp
    : public mlir::OpConversionPattern<libtriton::dlpack::ToMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::ToMemRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    mlir::MemRefType memRefType =
        mlir::cast<mlir::MemRefType>(op.getOutput().getType());
    const std::uint32_t rank = static_cast<std::uint32_t>(memRefType.getRank());

    mlir::Type outputType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!outputType)
      return mlir::failure();

    mlir::Type convertedDLContextType =
        libtriton::conversion::utils::DLContextLLVMDescriptor::getLLVMType(
            context);
    mlir::Type convertedDLDataTypeType =
        libtriton::conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(
            context);
    mlir::LLVM::LLVMStructType dlContextTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLContextType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlContextTy || !dlDataTypeTy)
      return mlir::failure();

    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    libtriton::conversion::utils::DLTensorLLVMDescriptor dlTensor =
        libtriton::conversion::utils::DLTensorLLVMDescriptor::from(
            adaptor.getInput());
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
    for (std::uint32_t i = 0; i < rank; ++i) {
      memDesc.setSize(rewriter, loc, i, shapeValues[i]);
      memDesc.setStride(rewriter, loc, i, strideValues[i]);
    }

    rewriter.replaceOp(op, mlir::Value(memDesc));
    return mlir::success();
  }
};

struct LowerTensorFromLLVMOp
    : public mlir::OpConversionPattern<libtriton::dlpack::TensorFromLLVMOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::TensorFromLLVMOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type convertedTensorType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!convertedTensorType ||
        adaptor.getInput().getType() != convertedTensorType)
      return mlir::failure();
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

struct LowerTensorToLLVMOp
    : public mlir::OpConversionPattern<libtriton::dlpack::TensorToLLVMOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::TensorToLLVMOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getInput().getType() != op.getOutput().getType())
      return mlir::failure();
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

void populateDLPackToLLVMTypeConversions(
    mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([](libtriton::dlpack::DLContextType type) {
    return libtriton::conversion::utils::DLContextLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  typeConverter.addConversion([](libtriton::dlpack::DLDataTypeType type) {
    return libtriton::conversion::utils::DLDataTypeLLVMDescriptor::getLLVMType(
        type.getContext());
  });
  typeConverter.addConversion([&](libtriton::dlpack::DLTensorType type)
                                  -> mlir::Type {
    return libtriton::conversion::utils::DLTensorLLVMDescriptor::getLLVMType(
        type.getContext(), typeConverter.getPointerBitwidth());
  });
  typeConverter.addConversion(
      [&](libtriton::dlpack::DLManagedTensorType type) -> mlir::Type {
        return libtriton::conversion::utils::DLManagedTensorLLVMDescriptor::
            getLLVMType(type.getContext(), typeConverter.getPointerBitwidth());
      });
}

void populateDLPackToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  mlir::MLIRContext *context = patterns.getContext();
  populateDLPackToLLVMTypeConversions(typeConverter);

  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, typeConverter);
  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<LowerFromMemRefOwnedOp, LowerFromMemRefBorrowedOp,
               LowerTensorFromLLVMOp, LowerTensorToLLVMOp, LowerViewOp,
               LowerToMemRefOp>(typeConverter, context);

  target.addIllegalDialect<libtriton::dlpack::DLPackDialect>();
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
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, libtriton::dlpack::DLPackDialect *dialect) {
        dialect->addInterfaces<DLPackToLLVMDialectInterface>();
      });
}

} // namespace libtriton::dlpack
