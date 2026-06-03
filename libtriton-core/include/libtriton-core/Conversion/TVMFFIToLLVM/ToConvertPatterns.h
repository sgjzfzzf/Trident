#ifndef LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TOCONVERTPATTERNS_H_
#define LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TOCONVERTPATTERNS_H_

#include <cstdint>
#include <optional>

#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFILLVMDescriptors.h"
#include "libtriton-core/Conversion/Utils/IConstantUtils.h"
#include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "tvm/ffi/c_api.h"

namespace libtriton::tvm_ffi {

template <typename SourceType, typename TargetType> struct ConverterBase {
  static bool match(mlir::Type src, mlir::Type tgt) {
    return mlir::isa<SourceType>(src) && mlir::isa<TargetType>(tgt);
  }
};

std::optional<mlir::Value>
buildAny(mlir::OpBuilder &builder, mlir::Location loc,
         const mlir::TypeConverter *typeConverter, mlir::Type anyType,
         mlir::TypedValue<mlir::IntegerType> typeIndexValue,
         mlir::TypedValue<mlir::IntegerType> payloadBitsValue);

/// Convert a typed value to the LLVM struct representation of !tvm_ffi.any.
///
/// This is equivalent to the conversion performed by tvm_ffi.to when the
/// target type is !tvm_ffi.any, but does not require a ToOp.  The caller
/// provides the adapted (type-converted) \p adaptedInput together with the
/// \p originalInputType to select the right conversion strategy.
///
/// Returns the LLVM struct (!llvm.struct<(i32, i32, i64)>) that carries the
/// TVMFFIAny ABI representation, or std::nullopt on failure.
std::optional<mlir::Value>
toAnyStruct(mlir::Value adaptedInput, mlir::Type originalInputType,
            mlir::OpBuilder &builder, mlir::Location loc,
            const mlir::TypeConverter *typeConverter);

struct AnyToAnyPattern : ConverterBase<AnyType, AnyType> {
  static mlir::Value convert(ToOp /*op*/, ToOp::Adaptor adaptor,
                             mlir::OpBuilder & /*builder*/,
                             const mlir::TypeConverter * /*typeConverter*/);
};

struct AnyToDLTensorPattern : ConverterBase<AnyType, dlpack::DLTensorType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToFloat64Pattern : ConverterBase<AnyType, mlir::Float64Type> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToIntegerPattern : ConverterBase<AnyType, mlir::IntegerType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToObjectHandlePattern : ConverterBase<AnyType, ObjectHandleType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct AnyToPointerPattern
    : ConverterBase<AnyType, mlir::LLVM::LLVMPointerType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct Float64ToAnyPattern : ConverterBase<mlir::Float64Type, AnyType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct IntegerToAnyPattern : ConverterBase<mlir::IntegerType, AnyType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct ObjectHandleToAnyPattern : ConverterBase<ObjectHandleType, AnyType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct ObjectHandleToDLTensorPattern
    : ConverterBase<ObjectHandleType, dlpack::DLTensorType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

struct PointerToAnyPattern
    : ConverterBase<mlir::LLVM::LLVMPointerType, AnyType> {
  static mlir::Value convert(ToOp op, ToOp::Adaptor adaptor,
                             mlir::OpBuilder &builder,
                             const mlir::TypeConverter *typeConverter);
};

template <typename... Patterns> struct PatternSet {
  static bool supports(mlir::Type sourceType, mlir::Type targetType) {
    return (Patterns::match(sourceType, targetType) || ...);
  }

  static std::optional<mlir::Value>
  convert(ToOp op, ToOp::Adaptor adaptor, mlir::OpBuilder &builder,
          const mlir::TypeConverter *typeConverter) {
    std::optional<mlir::Value> convertedValue;
    const bool matched =
        ((Patterns::match(op.getInput().getType(), op.getOutput().getType()) &&
          (convertedValue =
               Patterns::convert(op, adaptor, builder, typeConverter),
           true)) ||
         ...);
    if (matched) {
      return convertedValue;
    } else {
      return std::nullopt;
    }
  }
};

using ToPatternSet =
    PatternSet<AnyToAnyPattern, AnyToDLTensorPattern, AnyToFloat64Pattern,
               AnyToIntegerPattern, AnyToObjectHandlePattern,
               AnyToPointerPattern, Float64ToAnyPattern, IntegerToAnyPattern,
               ObjectHandleToAnyPattern, ObjectHandleToDLTensorPattern,
               PointerToAnyPattern>;

} // namespace libtriton::tvm_ffi

#endif // LIBTRITON_CORE_CONVERSION_TVMFFITOLLVM_TOCONVERTPATTERNS_H_
