#ifndef LIBTRITON_CORE_CONVERSION_UTILS_ICONSTANTUTILS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_ICONSTANTUTILS_H_

#include <cstdint>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::conversion::utils {

mlir::TypedValue<mlir::IntegerType> emitI8Constant(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::MLIRContext *context,
                                                   int64_t value);

mlir::TypedValue<mlir::IntegerType> emitI16Constant(mlir::OpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::MLIRContext *context,
                                                    int64_t value);

mlir::TypedValue<mlir::IntegerType> emitI32Constant(mlir::OpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::MLIRContext *context,
                                                    int64_t value);

mlir::TypedValue<mlir::IntegerType> emitI64Constant(mlir::OpBuilder &builder,
                                                    mlir::Location loc,
                                                    mlir::MLIRContext *context,
                                                    int64_t value);

mlir::TypedValue<mlir::IntegerType>
castIntegerTo(mlir::OpBuilder &builder, mlir::Location loc,
              mlir::IntegerType targetType,
              mlir::TypedValue<mlir::IntegerType> integerValue);

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_ICONSTANTUTILS_H_
