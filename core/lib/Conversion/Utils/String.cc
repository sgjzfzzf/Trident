//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Utils/String.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

namespace trident::conversion::utils {

mlir::Value getString(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef content) {
  mlir::MLIRContext *context = builder.getContext();
  const mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(context);
  const mlir::Type i8Ty = mlir::IntegerType::get(context, 8);
  const mlir::Value length = mlir::LLVM::ConstantOp::create(
      builder, loc, mlir::IntegerType::get(context, 64),
      static_cast<int64_t>(content.size() + 1));
  const mlir::Value stringPtr =
      mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, i8Ty, length);
  for (auto [index, character] : llvm::enumerate(llvm::concat<int8_t>(
           llvm::map_range(
               content,
               [](char character) { return static_cast<int8_t>(character); }),
           llvm::ArrayRef<int8_t>{0}))) {
    const mlir::Value elementPtr = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i8Ty, stringPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
    mlir::LLVM::StoreOp::create(
        builder, loc,
        mlir::LLVM::ConstantOp::create(builder, loc, i8Ty, character),
        elementPtr);
  }
  return stringPtr;
}

} // namespace trident::conversion::utils
