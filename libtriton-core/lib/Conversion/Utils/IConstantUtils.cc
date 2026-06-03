// IConstantUtils.cc - Implementation of integer constant utilities.

#include "libtriton-core/Conversion/Utils/IConstantUtils.h"

namespace libtriton::conversion::utils {

template <uint32_t BitWidth>
mlir::TypedValue<mlir::IntegerType>
emitIConstant(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              mlir::MLIRContext *context, int64_t value) {
  const mlir::Type iTy = mlir::IntegerType::get(context, BitWidth);
  return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
      mlir::LLVM::ConstantOp::create(rewriter, loc, iTy, value).getResult());
}

mlir::TypedValue<mlir::IntegerType>
emitI8Constant(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
               mlir::MLIRContext *context, int64_t value) {
  return emitIConstant<8>(rewriter, loc, context, value);
}

mlir::TypedValue<mlir::IntegerType>
emitI16Constant(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                mlir::MLIRContext *context, int64_t value) {
  return emitIConstant<16>(rewriter, loc, context, value);
}

mlir::TypedValue<mlir::IntegerType>
emitI32Constant(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                mlir::MLIRContext *context, int64_t value) {
  return emitIConstant<32>(rewriter, loc, context, value);
}

mlir::TypedValue<mlir::IntegerType>
emitI64Constant(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                mlir::MLIRContext *context, int64_t value) {
  return emitIConstant<64>(rewriter, loc, context, value);
}

mlir::TypedValue<mlir::IntegerType>
castIntegerTo(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              mlir::IntegerType targetType,
              mlir::TypedValue<mlir::IntegerType> integerValue) {
  const int32_t srcWidth = integerValue.getType().getWidth();
  const int32_t tgtWidth = targetType.getWidth();
  if (srcWidth < tgtWidth) {
    return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
        mlir::LLVM::ZExtOp::create(rewriter, loc, targetType, integerValue)
            .getResult());
  } else if (srcWidth > tgtWidth) {
    return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
        mlir::LLVM::TruncOp::create(rewriter, loc, targetType, integerValue)
            .getResult());
  } else {
    return integerValue;
  }
}

} // namespace libtriton::conversion::utils
