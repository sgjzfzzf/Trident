// DLPackToLLVM.cc - Pass that lowers DLPack dialect ops to LLVM dialect.

#include <algorithm>
#include <cstdint>
#include <tuple>

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tvm_ffi_bindings/Conversion/DLPackToLLVM/DLPackToLLVM.h"
#include "tvm_ffi_bindings/Conversion/Utils/DLPackLLVMDescriptors.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackOps.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::dlpack {
namespace {

mlir::Value materializeCast(mlir::OpBuilder &builder, mlir::Type resultType,
                            mlir::ValueRange inputs, mlir::Location loc) {
  if (inputs.size() != 1)
    return {};
  return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs)
      .getResult(0);
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyShapeFromMemRefDescriptorToArray(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::MemRefDescriptor memRefDescriptor,
                                     const std::uint32_t rank, mlir::Type i64Ty,
                                     mlir::Type ptrTy) {
  const std::uint32_t allocCount = std::max<std::uint32_t>(rank, 1);
  const mlir::Value sizeVal = mlir::LLVM::ConstantOp::create(
      rewriter, loc, i64Ty, static_cast<int64_t>(allocCount));
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeAlloca =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, sizeVal)
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
copyStrideFromMemRefDescriptorToArray(mlir::ConversionPatternRewriter &rewriter,
                                      mlir::Location loc,
                                      mlir::MemRefDescriptor memRefDescriptor,
                                      const std::uint32_t rank,
                                      mlir::Type i64Ty, mlir::Type ptrTy) {
  const std::uint32_t allocCount = std::max<std::uint32_t>(rank, 1);
  const mlir::Value sizeVal = mlir::LLVM::ConstantOp::create(
      rewriter, loc, i64Ty, static_cast<int64_t>(allocCount));
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesAlloca =
      mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
          mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, i64Ty, sizeVal)
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

struct LowerFromMemRefOp
    : public mlir::OpConversionPattern<libtriton::dlpack::FromMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::FromMemRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    const mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

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

    mlir::Type convertedTensorType =
        llvmTypeConverter->convertType(op.getOutput().getType());
    mlir::Type convertedDLContextType = getDLContextLLVMType(context);
    mlir::Type convertedDLDataTypeType = getDLDataTypeLLVMType(context);

    mlir::LLVM::LLVMStructType dlTensorTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedTensorType);
    mlir::LLVM::LLVMStructType dlContextTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLContextType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlTensorTy || !dlContextTy || !dlDataTypeTy)
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
    mlir::TypedValue<mlir::IntegerType> elemOffset =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            memDesc.offset(rewriter, loc));

    mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapeAlloca =
        copyShapeFromMemRefDescriptorToArray(rewriter, loc, memDesc, rank,
                                             i64Ty, ptrTy);
    mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesAlloca =
        copyStrideFromMemRefDescriptorToArray(rewriter, loc, memDesc, rank,
                                              i64Ty, ptrTy);

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
            dlDataType, shapeAlloca, stridesAlloca, elemOffset);

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

    mlir::Type convertedDLContextType = getDLContextLLVMType(context);
    mlir::Type convertedDLDataTypeType = getDLDataTypeLLVMType(context);
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

class ConvertDLPackToLLVMPass
    : public mlir::PassWrapper<ConvertDLPackToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertDLPackToLLVMPass)

  llvm::StringRef getArgument() const final { return "convert-dlpack-to-llvm"; }

  llvm::StringRef getDescription() const final {
    return "Lower DLPack dialect operations to LLVM dialect";
  }

  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    populateDLPackToLLVMTypeConversions(typeConverter);

    mlir::RewritePatternSet patterns(&context);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    patterns.add<LowerFromMemRefOp, LowerToMemRefOp>(typeConverter, &context);

    mlir::ConversionTarget target(context);
    target.addIllegalDialect<libtriton::dlpack::DLPackDialect>();
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
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    if (failed(mlir::applyPartialConversion(getOperation(), target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static mlir::PassRegistration<ConvertDLPackToLLVMPass> kPass;

} // namespace

mlir::LLVM::LLVMStructType getDLContextLLVMType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {mlir::IntegerType::get(context, 32),
                mlir::IntegerType::get(context, 32)});
}

mlir::LLVM::LLVMStructType getDLDataTypeLLVMType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {mlir::IntegerType::get(context, 8), mlir::IntegerType::get(context, 8),
       mlir::IntegerType::get(context, 16)});
}

mlir::LLVM::LLVMStructType getDLTensorLLVMType(mlir::MLIRContext *context,
                                               std::uint32_t sizeTWidth) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      context,
      {mlir::LLVM::LLVMPointerType::get(context), getDLContextLLVMType(context),
       mlir::IntegerType::get(context, 32), getDLDataTypeLLVMType(context),
       mlir::LLVM::LLVMPointerType::get(context),
       mlir::LLVM::LLVMPointerType::get(context),
       mlir::IntegerType::get(context, sizeTWidth)});
}

void populateDLPackToLLVMTypeConversions(
    mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([](libtriton::dlpack::DLContextType type) {
    return getDLContextLLVMType(type.getContext());
  });
  typeConverter.addConversion([](libtriton::dlpack::DLDataTypeType type) {
    return getDLDataTypeLLVMType(type.getContext());
  });
  typeConverter.addConversion(
      [&](libtriton::dlpack::DLTensorTypeType type) -> mlir::Type {
        mlir::Type convertedDLContextType =
            getDLContextLLVMType(type.getContext());
        mlir::Type convertedDLDataTypeType =
            getDLDataTypeLLVMType(type.getContext());

        mlir::LLVM::LLVMStructType dlContextTy =
            mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLContextType);
        mlir::LLVM::LLVMStructType dlDataTypeTy =
            mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
        if (!dlContextTy || !dlDataTypeTy)
          return mlir::Type();

        return mlir::LLVM::LLVMStructType::getLiteral(
            type.getContext(),
            {mlir::LLVM::LLVMPointerType::get(type.getContext()), dlContextTy,
             mlir::IntegerType::get(type.getContext(), 32), dlDataTypeTy,
             mlir::LLVM::LLVMPointerType::get(type.getContext()),
             mlir::LLVM::LLVMPointerType::get(type.getContext()),
             mlir::IntegerType::get(type.getContext(),
                                    typeConverter.getPointerBitwidth())});
      });
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);
}

std::unique_ptr<mlir::Pass> createConvertDLPackToLLVMPass() {
  return std::make_unique<ConvertDLPackToLLVMPass>();
}

void registerConvertDLPackToLLVMPass() {
  // Registration is handled by static PassRegistration above.
}

void registerDLPackToLLVMPasses() { registerConvertDLPackToLLVMPass(); }

} // namespace libtriton::dlpack
