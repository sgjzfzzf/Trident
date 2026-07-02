//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_BACKENDTYPECONVERSION_H_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_TRANSFORMS_BACKENDTYPECONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace libtriton::torch {

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

} // namespace libtriton::torch

#endif // LIBTRITON_CORE_DIALET_TORCHEXT_TRANSFORMS_BACKENDTYPECONVERSION_H_
