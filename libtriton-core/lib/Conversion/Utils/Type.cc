//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/Utils/Type.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"

// ---------------------------------------------------------------------------
// DLPack LLVM struct type helpers
// ---------------------------------------------------------------------------

mlir::LLVM::LLVMStructType
libtriton::conversion::utils::getDLDeviceType(mlir::MLIRContext *context) {
  mlir::IntegerType i32Ty = mlir::IntegerType::get(context, 32);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty},
                                                /*packed=*/false);
}

mlir::LLVM::LLVMStructType
libtriton::conversion::utils::getDLDataType(mlir::MLIRContext *context) {
  mlir::IntegerType i8Ty = mlir::IntegerType::get(context, 8);
  mlir::IntegerType i16Ty = mlir::IntegerType::get(context, 16);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i8Ty, i8Ty, i16Ty},
                                                /*packed=*/false);
}

mlir::LLVM::LLVMStructType
libtriton::conversion::utils::getDLTensorType(mlir::MLIRContext *context) {
  mlir::IntegerType i32Ty = mlir::IntegerType::get(context, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(context, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {ptrTy, getDLDeviceType(context), i32Ty, getDLDataType(context),
                ptrTy, ptrTy, i64Ty});
}

mlir::LLVM::LLVMStructType
libtriton::conversion::utils::getTVMFFIAnyType(mlir::MLIRContext *context) {
  mlir::IntegerType i32Ty = mlir::IntegerType::get(context, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(context, 64);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i64Ty});
}
