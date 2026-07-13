//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_BACKENDTYPECONVERSION_H_
#define TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_BACKENDTYPECONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace trident::torch {

/// Populate function callback type conversion patterns and set up legality.
/// This handles func::FuncOp, func::CallOp, control flow structural
/// conversions, return ops, and marks unknown ops dynamically legal with
/// respect to branch/return type conversion.
void populateFuncBackendTypeConversionPatterns(
    mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target);

/// Set up the provided ConversionTarget and LLVMTypeConverter for converting
/// from Torch dialect types to LLVM types along the backend boundary.
/// Currently handles:
///   - Torch tensor types (ValueTensorType, NonValueTensorType) -> llvm.ptr
///   - Torch BoolType -> i1
///   - Torch DeviceType -> struct<i32, i32>
///   - Torch IntType -> i64
///   - Torch FloatType -> f64
///   - Torch StringType -> llvm.ptr
///   - Torch OptionalType -> llvm.struct
///   - Torch ListType -> llvm.ptr
///   - Torch TupleType -> llvm.struct
/// Generator conversion is not implemented yet.
void setupBackendTypeConversion(mlir::ConversionTarget &target,
                                mlir::TypeConverter &typeConverter);

} // namespace trident::torch

#endif // TRIDENT_CORE_DIALET_TORCHEXT_TRANSFORMS_BACKENDTYPECONVERSION_H_
