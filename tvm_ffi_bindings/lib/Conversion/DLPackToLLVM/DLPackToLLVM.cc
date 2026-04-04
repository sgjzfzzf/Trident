// DLPackToLLVM.cc - Pass that lowers DLPack dialect ops to LLVM dialect.

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
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackDialect.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackOps.h"
#include "tvm_ffi_bindings/Dialect/DLPack/IR/DLPackTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace libtriton::dlpack {
namespace {

mlir::Type getOpaquePtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(context);
}

mlir::Type getIntegerType(mlir::MLIRContext *context, std::uint32_t width) {
  return mlir::IntegerType::get(context, width);
}

mlir::Value materializeCast(mlir::OpBuilder &builder, mlir::Type resultType,
                            mlir::ValueRange inputs, mlir::Location loc) {
  if (inputs.size() != 1)
    return {};
  return mlir::UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs)
      .getResult(0);
}

mlir::Type
getConvertedDLContextType(const mlir::LLVMTypeConverter &typeConverter,
                          mlir::MLIRContext *context) {
  return typeConverter.convertType(
      libtriton::dlpack::DLContextType::get(context));
}

mlir::Type
getConvertedDLDataTypeType(const mlir::LLVMTypeConverter &typeConverter,
                           mlir::MLIRContext *context) {
  return typeConverter.convertType(
      libtriton::dlpack::DLDataTypeType::get(context));
}

mlir::Value buildDLContextValue(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc, mlir::Type i32Ty,
                                mlir::LLVM::LLVMStructType dlContextTy) {
  mlir::Value dlContextValue =
      mlir::LLVM::PoisonOp::create(rewriter, loc, dlContextTy);
  dlContextValue = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlContextValue,
      mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                     static_cast<std::int32_t>(kDLCPU)),
      llvm::ArrayRef<int64_t>{0});
  return mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlContextValue,
      mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0LL),
      llvm::ArrayRef<int64_t>{1});
}

mlir::Value buildDLDataTypeValue(mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Location loc, mlir::Type i8Ty,
                                 mlir::Type i16Ty,
                                 mlir::LLVM::LLVMStructType dlDataTypeTy,
                                 std::uint8_t dtypeCode, std::uint8_t dtypeBits,
                                 std::uint16_t dtypeLanes) {
  mlir::Value dlDataTypeValue =
      mlir::LLVM::PoisonOp::create(rewriter, loc, dlDataTypeTy);
  dlDataTypeValue = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlDataTypeValue,
      mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                     static_cast<int64_t>(dtypeCode)),
      llvm::ArrayRef<int64_t>{0});
  dlDataTypeValue = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlDataTypeValue,
      mlir::LLVM::ConstantOp::create(rewriter, loc, i8Ty,
                                     static_cast<int64_t>(dtypeBits)),
      llvm::ArrayRef<int64_t>{1});
  return mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlDataTypeValue,
      mlir::LLVM::ConstantOp::create(rewriter, loc, i16Ty,
                                     static_cast<int64_t>(dtypeLanes)),
      llvm::ArrayRef<int64_t>{2});
}

mlir::Value
buildDLTensorValue(mlir::ConversionPatternRewriter &rewriter,
                   mlir::Location loc, mlir::LLVM::LLVMStructType dlTensorTy,
                   mlir::TypedValue<mlir::LLVM::LLVMPointerType> dataPtr,
                   mlir::Value dlContextValue,
                   mlir::TypedValue<mlir::IntegerType> ndimValue,
                   mlir::Value dlDataTypeValue,
                   mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapePtr,
                   mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesPtr,
                   mlir::TypedValue<mlir::IntegerType> byteOffset) {
  mlir::Value dlTensor =
      mlir::LLVM::PoisonOp::create(rewriter, loc, dlTensorTy);
  dlTensor = mlir::LLVM::InsertValueOp::create(rewriter, loc, dlTensor, dataPtr,
                                               llvm::ArrayRef<int64_t>{0});
  dlTensor = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlTensor, dlContextValue, llvm::ArrayRef<int64_t>{1});
  dlTensor = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlTensor, ndimValue, llvm::ArrayRef<int64_t>{2});
  dlTensor = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlTensor, dlDataTypeValue, llvm::ArrayRef<int64_t>{3});
  dlTensor = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlTensor, shapePtr, llvm::ArrayRef<int64_t>{4});
  dlTensor = mlir::LLVM::InsertValueOp::create(
      rewriter, loc, dlTensor, stridesPtr, llvm::ArrayRef<int64_t>{5});
  return mlir::LLVM::InsertValueOp::create(rewriter, loc, dlTensor, byteOffset,
                                           llvm::ArrayRef<int64_t>{6});
}

struct DLTensorLLVMView {
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> dataPtr;
  mlir::Value dlContext;
  mlir::TypedValue<mlir::IntegerType> ndim;
  mlir::Value dlDataType;
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> shapePtr;
  mlir::TypedValue<mlir::LLVM::LLVMPointerType> stridesPtr;
  mlir::TypedValue<mlir::IntegerType> byteOffset;
};

DLTensorLLVMView unpackDLTensorValue(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::Value dltensorValue) {
  DLTensorLLVMView view;
  view.dataPtr = mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
      mlir::LLVM::ExtractValueOp::create(rewriter, loc, dltensorValue,
                                         llvm::ArrayRef<int64_t>{0})
          .getResult());
  view.dlContext = mlir::LLVM::ExtractValueOp::create(
      rewriter, loc, dltensorValue, llvm::ArrayRef<int64_t>{1});
  view.ndim = mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
      mlir::LLVM::ExtractValueOp::create(rewriter, loc, dltensorValue,
                                         llvm::ArrayRef<int64_t>{2})
          .getResult());
  view.dlDataType = mlir::LLVM::ExtractValueOp::create(
      rewriter, loc, dltensorValue, llvm::ArrayRef<int64_t>{3});
  view.shapePtr = mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
      mlir::LLVM::ExtractValueOp::create(rewriter, loc, dltensorValue,
                                         llvm::ArrayRef<int64_t>{4})
          .getResult());
  view.stridesPtr = mlir::cast<mlir::TypedValue<mlir::LLVM::LLVMPointerType>>(
      mlir::LLVM::ExtractValueOp::create(rewriter, loc, dltensorValue,
                                         llvm::ArrayRef<int64_t>{5})
          .getResult());
  view.byteOffset = mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
      mlir::LLVM::ExtractValueOp::create(rewriter, loc, dltensorValue,
                                         llvm::ArrayRef<int64_t>{6})
          .getResult());
  return view;
}

mlir::TypedValue<mlir::LLVM::LLVMPointerType>
copyShapeFromMemRefDescriptorToArray(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::MemRefDescriptor memRefDescriptor,
                                     std::uint32_t rank, mlir::Type i64Ty,
                                     mlir::Type ptrTy) {
  std::uint32_t allocCount = std::max<std::uint32_t>(rank, 1);
  mlir::Value sizeVal = mlir::LLVM::ConstantOp::create(
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
                                      std::uint32_t rank, mlir::Type i64Ty,
                                      mlir::Type ptrTy) {
  std::uint32_t allocCount = std::max<std::uint32_t>(rank, 1);
  mlir::Value sizeVal = mlir::LLVM::ConstantOp::create(
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
    std::uint32_t rank, mlir::Type i64Ty, mlir::Type ptrTy) {
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
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    mlir::MemRefType memRefType =
        mlir::cast<mlir::MemRefType>(op.getInput().getType());
    std::uint32_t rank = static_cast<std::uint32_t>(memRefType.getRank());

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
    mlir::Type convertedDLContextType =
        getConvertedDLContextType(*llvmTypeConverter, context);
    mlir::Type convertedDLDataTypeType =
        getConvertedDLDataTypeType(*llvmTypeConverter, context);

    mlir::LLVM::LLVMStructType dlTensorTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedTensorType);
    mlir::LLVM::LLVMStructType dlContextTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLContextType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlTensorTy || !dlContextTy || !dlDataTypeTy)
      return mlir::failure();

    // Determine DLDataType fields from the memref element type
    uint8_t dtypeCode;
    uint8_t dtypeBits;
    uint16_t dtypeLanes = 1;
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
      dtypeBits = static_cast<uint8_t>(integerType.getWidth());
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
    mlir::Value dlContextValue =
        buildDLContextValue(rewriter, loc, i32Ty, dlContextTy);

    // Build DLDataType struct: {code, bits, lanes = 1}
    mlir::Value dlDataTypeValue =
        buildDLDataTypeValue(rewriter, loc, i8Ty, i16Ty, dlDataTypeTy,
                             dtypeCode, dtypeBits, dtypeLanes);

    mlir::TypedValue<mlir::IntegerType> ndimValue =
        mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
            mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           static_cast<int64_t>(rank))
                .getResult());

    mlir::Value dlTensor = buildDLTensorValue(
        rewriter, loc, dlTensorTy, dataPtr, dlContextValue, ndimValue,
        dlDataTypeValue, shapeAlloca, stridesAlloca, elemOffset);

    rewriter.replaceOp(op, dlTensor);
    return mlir::success();
  }
};

struct LowerToMemRefOp
    : public mlir::OpConversionPattern<libtriton::dlpack::ToMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(libtriton::dlpack::ToMemRefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = op.getContext();

    mlir::MemRefType memRefType =
        mlir::cast<mlir::MemRefType>(op.getOutput().getType());
    std::uint32_t rank = static_cast<std::uint32_t>(memRefType.getRank());

    mlir::Type outputType =
        getTypeConverter()->convertType(op.getOutput().getType());
    if (!outputType)
      return mlir::failure();

    const mlir::LLVMTypeConverter *llvmTypeConverter =
        static_cast<const mlir::LLVMTypeConverter *>(getTypeConverter());
    mlir::Type convertedDLContextType =
        getConvertedDLContextType(*llvmTypeConverter, context);
    mlir::Type convertedDLDataTypeType =
        getConvertedDLDataTypeType(*llvmTypeConverter, context);
    mlir::LLVM::LLVMStructType dlContextTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLContextType);
    mlir::LLVM::LLVMStructType dlDataTypeTy =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
    if (!dlContextTy || !dlDataTypeTy)
      return mlir::failure();

    mlir::Type i64Ty = mlir::IntegerType::get(context, 64);
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);

    DLTensorLLVMView dlTensorView =
        unpackDLTensorValue(rewriter, loc, adaptor.getInput());

    // Build an LLVM memref descriptor from the DLTensor fields
    mlir::MemRefDescriptor memDesc =
        mlir::MemRefDescriptor::poison(rewriter, loc, outputType);
    memDesc.setAllocatedPtr(rewriter, loc, dlTensorView.dataPtr);
    memDesc.setAlignedPtr(rewriter, loc, dlTensorView.dataPtr);
    memDesc.setOffset(rewriter, loc, dlTensorView.byteOffset);

    std::tuple<llvm::SmallVector<mlir::Value>, llvm::SmallVector<mlir::Value>>
        shapeAndStrideValues = extractShapeAndStrideFromArrays(
            rewriter, loc, dlTensorView.shapePtr, dlTensorView.stridesPtr, rank,
            i64Ty, ptrTy);
    llvm::SmallVector<mlir::Value> &shapeValues =
        std::get<0>(shapeAndStrideValues);
    llvm::SmallVector<mlir::Value> &strideValues =
        std::get<1>(shapeAndStrideValues);
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
      context, {getIntegerType(context, 32), getIntegerType(context, 32)});
}

mlir::LLVM::LLVMStructType getDLDataTypeLLVMType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {getIntegerType(context, 8), getIntegerType(context, 8),
                getIntegerType(context, 16)});
}

mlir::LLVM::LLVMStructType getDLTensorLLVMType(mlir::MLIRContext *context,
                                               std::uint32_t sizeTWidth) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {getOpaquePtrType(context), getDLContextLLVMType(context),
                getIntegerType(context, 32), getDLDataTypeLLVMType(context),
                getOpaquePtrType(context), getOpaquePtrType(context),
                getIntegerType(context, sizeTWidth)});
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
            getConvertedDLContextType(typeConverter, type.getContext());
        mlir::Type convertedDLDataTypeType =
            getConvertedDLDataTypeType(typeConverter, type.getContext());

        mlir::LLVM::LLVMStructType dlContextTy =
            mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLContextType);
        mlir::LLVM::LLVMStructType dlDataTypeTy =
            mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedDLDataTypeType);
        if (!dlContextTy || !dlDataTypeTy)
          return mlir::Type();

        return mlir::LLVM::LLVMStructType::getLiteral(
            type.getContext(),
            {getOpaquePtrType(type.getContext()), dlContextTy,
             getIntegerType(type.getContext(), 32), dlDataTypeTy,
             getOpaquePtrType(type.getContext()),
             getOpaquePtrType(type.getContext()),
             getIntegerType(type.getContext(),
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
