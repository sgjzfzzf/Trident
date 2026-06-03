#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TORULES_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TORULES_H_

#include <cstdint>
#include <optional>

#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFILLVMDescriptors.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "tvm/ffi/c_api.h"

namespace libtriton::tvm_ffi::to {

namespace convert {

template <typename SourceType, typename TargetType> struct ConverterBase {
  static bool match(mlir::Type src, mlir::Type tgt) {
    return mlir::isa<SourceType>(src) && mlir::isa<TargetType>(tgt);
  }
};

template <unsigned BitWidth>
mlir::TypedValue<mlir::IntegerType>
emitIConstant(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              mlir::MLIRContext *context, int64_t value) {
  const mlir::Type iTy = mlir::IntegerType::get(context, BitWidth);
  return mlir::cast<mlir::TypedValue<mlir::IntegerType>>(
      mlir::LLVM::ConstantOp::create(rewriter, loc, iTy, value).getResult());
}

std::optional<mlir::Value>
buildAny(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
         const mlir::TypeConverter *typeConverter, mlir::Type anyType,
         mlir::TypedValue<mlir::IntegerType> typeIndexValue,
         mlir::TypedValue<mlir::IntegerType> payloadBitsValue);

mlir::TypedValue<mlir::IntegerType>
castIntegerToI64(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 mlir::Type i64Ty,
                 mlir::TypedValue<mlir::IntegerType> integerValue);

struct AnyToAnyConverter : ConverterBase<AnyType, AnyType> {
  static mlir::Value convert(ToOp /*op*/, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter & /*rewriter*/,
                             const mlir::TypeConverter * /*typeConverter*/);
};

struct ObjectHandleToAnyConverter : ConverterBase<ObjectHandleType, AnyType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct PointerToAnyConverter
    : ConverterBase<mlir::LLVM::LLVMPointerType, AnyType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct Float64ToAnyConverter : ConverterBase<mlir::Float64Type, AnyType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct IntegerToAnyConverter : ConverterBase<mlir::IntegerType, AnyType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToDLTensorConverter : ConverterBase<AnyType, dlpack::DLTensorType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToFloat64Converter : ConverterBase<AnyType, mlir::Float64Type> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToObjectHandleConverter : ConverterBase<AnyType, ObjectHandleType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToPointerConverter
    : ConverterBase<AnyType, mlir::LLVM::LLVMPointerType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToIntegerConverter : ConverterBase<AnyType, mlir::IntegerType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

struct ObjectHandleToDLTensorConverter
    : ConverterBase<ObjectHandleType, dlpack::DLTensorType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::ConversionPatternRewriter &rewriter,
                             const mlir::TypeConverter *typeConverter);
};

} // namespace convert

template <typename... Rules> struct RuleSet {
  static bool supports(mlir::Type sourceType, mlir::Type targetType) {
    return (Rules::match(sourceType, targetType) || ...);
  }

  static std::optional<mlir::Value>
  convert(ToOp op, ToOp::Adaptor adaptor,
          mlir::ConversionPatternRewriter &rewriter,
          const mlir::TypeConverter *typeConverter) {
    std::optional<mlir::Value> convertedValue;
    const bool matched =
        ((Rules::match(op.getInput().getType(), op.getOutput().getType()) &&
          (convertedValue =
               Rules::convert(op, adaptor, rewriter, typeConverter),
           true)) ||
         ...);
    if (matched) {
      return convertedValue;
    } else {
      return std::nullopt;
    }
  }
};

using AnyToAnyRule = convert::AnyToAnyConverter;
using AnyToDLTensorRule = convert::AnyToDLTensorConverter;
using AnyToFloat64Rule = convert::AnyToFloat64Converter;
using AnyToIntegerRule = convert::AnyToIntegerConverter;
using AnyToObjectHandleRule = convert::AnyToObjectHandleConverter;
using AnyToPointerRule = convert::AnyToPointerConverter;
using Float64ToAnyRule = convert::Float64ToAnyConverter;
using IntegerToAnyRule = convert::IntegerToAnyConverter;
using ObjectHandleToAnyRule = convert::ObjectHandleToAnyConverter;
using ObjectHandleToDLTensorRule = convert::ObjectHandleToDLTensorConverter;
using PointerToAnyRule = convert::PointerToAnyConverter;

using ToRuleSet =
    RuleSet<AnyToAnyRule, AnyToDLTensorRule, AnyToFloat64Rule, AnyToIntegerRule,
            AnyToObjectHandleRule, AnyToPointerRule, Float64ToAnyRule,
            IntegerToAnyRule, ObjectHandleToAnyRule, ObjectHandleToDLTensorRule,
            PointerToAnyRule>;

} // namespace libtriton::tvm_ffi::to

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TORULES_H_
