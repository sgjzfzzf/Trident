//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

namespace trident::dlpack {

mlir::LLVM::LLVMStructType
DLDeviceType::getLLVMType(mlir::MLIRContext *context) {
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(context, 32);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty});
}

mlir::LLVM::LLVMStructType
DLDataTypeType::getLLVMType(mlir::MLIRContext *context) {
  mlir::IntegerType const i8Ty = mlir::IntegerType::get(context, 8);
  mlir::IntegerType const i16Ty = mlir::IntegerType::get(context, 16);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i8Ty, i8Ty, i16Ty});
}

mlir::LLVM::LLVMStructType
DLTensorType::getLLVMType(mlir::MLIRContext *context) {
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(context, 32);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(context, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(context);
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {ptrTy, DLDeviceType::getLLVMType(context), i32Ty,
                DLDataTypeType::getLLVMType(context), ptrTy, ptrTy, i64Ty});
}

} // namespace trident::dlpack
