//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_TYPE_H_
#define TRIDENT_CORE_CONVERSION_UTILS_TYPE_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace trident::conversion::utils {

/// Returns the packed LLVM struct type matching DLDevice:
///   { i32 device_type, i32 device_id }  (8 bytes)
mlir::LLVM::LLVMStructType getDLDeviceType(mlir::MLIRContext *context);

/// Returns the packed LLVM struct type matching DLDataType:
///   { i8 code, i8 bits, i16 lanes }  (4 bytes)
mlir::LLVM::LLVMStructType getDLDataType(mlir::MLIRContext *context);

/// Returns the packed LLVM struct type matching DLTensor (48 bytes).
///
/// DLTensor layout (field indices -> offsets):
///   0: data        : ptr                               (offset  0)
///   1: device      : DLDevice    = {i32, i32}          (offset  8)
///   2: ndim        : i32                               (offset 16)
///   3: dtype       : DLDataType  = {i8, i8, i16}       (offset 20)
///   4: shape       : ptr                               (offset 24)
///   5: strides     : ptr                               (offset 32)
///   6: byte_offset : i64                               (offset 40)
mlir::LLVM::LLVMStructType getDLTensorType(mlir::MLIRContext *context);
/// Returns the LLVM struct type for TVMFFIAny:
///   { i32 type_index, i32 zero_padding, i64 payload }  (16 bytes)
mlir::LLVM::LLVMStructType getTVMFFIAnyType(mlir::MLIRContext *context);
} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_TYPE_H_
