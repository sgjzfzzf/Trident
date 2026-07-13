//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/GlobalString.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Type.h"

namespace trident::conversion::utils {

mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         llvm::ArrayRef<mlir::Value> args) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy =
      trident::conversion::utils::getTVMFFIAnyType(ctx);
  const size_t N = args.size();

  // Allocate contiguous args array and copy each pre-built slot.
  mlir::Value argsArray = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, N));
  for (auto [i, arg] : llvm::enumerate(args)) {
    mlir::Value dst =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, argsArray,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(builder, loc, anyTy, arg);
    mlir::LLVM::StoreOp::create(builder, loc, loaded, dst);
  }

  mlir::Value numArgs = mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, N);
  return callTVMFFIGlobalFunction(builder, loc, moduleOp, funcName, argsArray,
                                  numArgs);
}

mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         mlir::Value argsArray, mlir::Value numArgs) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

  mlir::LLVM::LLVMStructType byteArrayTy =
      mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  mlir::LLVM::LLVMStructType anyTy =
      trident::conversion::utils::getTVMFFIAnyType(ctx);

  // --- Step 1: TVMFFIByteArray { name_ptr, name_len } ---

  mlir::Value namePtr =
      getOrCreateGlobalString(builder, loc, moduleOp, funcName, funcName);
  mlir::Value nameSlot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, byteArrayTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::LLVM::StoreOp::create(
      builder, loc, namePtr,
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, byteArrayTy, nameSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
  mlir::LLVM::StoreOp::create(
      builder, loc,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, funcName.size()),
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, byteArrayTy, nameSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));

  // --- Step 2: TVMFFIFunctionGetGlobal ---

  mlir::Value funcSlot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, ptrTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::LLVM::LLVMFuncOp getGlobal =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIFunctionGetGlobal(moduleOp));
  mlir::LLVM::CallOp::create(builder, loc, getGlobal, {nameSlot, funcSlot});
  mlir::Value funcHandle =
      mlir::LLVM::LoadOp::create(builder, loc, ptrTy, funcSlot);

  // --- Step 3: Allocate and zero-initialize result slot ---

  mlir::Value zero32 = mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, 0);
  mlir::Value resultSlot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::LLVM::StoreOp::create(
      builder, loc, zero32,
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, resultSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
  mlir::LLVM::StoreOp::create(
      builder, loc, zero32,
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, resultSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));

  // --- Step 4: TVMFFIFunctionCall with runtime numArgs ---

  mlir::LLVM::LLVMFuncOp ffiCall =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIFunctionCall(moduleOp));
  mlir::LLVM::CallOp::create(builder, loc, ffiCall,
                             {funcHandle, argsArray, numArgs, resultSlot});

  // --- Step 5: TVMFFIObjectDecRef(funcHandle) ---

  mlir::LLVM::LLVMFuncOp decRef =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIObjectDecRef(moduleOp));
  mlir::LLVM::CallOp::create(builder, loc, decRef, {funcHandle});

  return resultSlot;
}

} // namespace trident::conversion::utils
