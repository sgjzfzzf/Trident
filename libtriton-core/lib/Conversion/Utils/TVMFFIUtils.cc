//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/Utils/TVMFFIUtils.h"
#include "libtriton-core/Conversion/Utils/Type.h"

namespace libtriton::conversion::utils {

mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         llvm::ArrayRef<mlir::Value> args) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

  mlir::LLVM::LLVMStructType byteArrayTy =
      mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  mlir::LLVM::LLVMStructType anyTy =
      libtriton::conversion::utils::getTVMFFIAnyType(ctx);
  const size_t N = args.size();

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

  // --- Step 2: TVMFFIFunctionGetGlobal → load function handle ---

  mlir::Value funcSlot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, ptrTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getGlobal =
      getOrCreateTVMFFIFunctionGetGlobal(moduleOp);
  if (mlir::failed(getGlobal)) {
    return mlir::failure();
  }
  mlir::LLVM::CallOp::create(builder, loc, *getGlobal, {nameSlot, funcSlot});
  mlir::Value funcHandle =
      mlir::LLVM::LoadOp::create(builder, loc, ptrTy, funcSlot);

  // --- Step 3: Copy pre-built slots into a contiguous args array ---

  mlir::Value argsArray = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, N));
  for (auto [i, arg] : llvm::enumerate(args)) {
    mlir::Value dst =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, argsArray,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
    // Load entire TVMFFIAny struct from caller's slot, store to contiguous
    // array — no field-by-field copy needed.
    mlir::Value loaded = mlir::LLVM::LoadOp::create(builder, loc, anyTy, arg);
    mlir::LLVM::StoreOp::create(builder, loc, loaded, dst);
  }

  // --- Step 4: Allocate result slot (kTVMFFINone=0) ---

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

  // --- Step 5: TVMFFIFunctionCall(func, args, N, &result) ---

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> call =
      getOrCreateTVMFFIFunctionCall(moduleOp);
  if (mlir::failed(call)) {
    return mlir::failure();
  }
  mlir::LLVM::CallOp::create(
      builder, loc, *call,
      {funcHandle, argsArray,
       mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, N), resultSlot});

  // --- Step 6: TVMFFIObjectDecRef(funcHandle) ---

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> decRef =
      getOrCreateTVMFFIObjectDecRef(moduleOp);
  if (mlir::failed(decRef)) {
    return mlir::failure();
  }
  mlir::LLVM::CallOp::create(builder, loc, *decRef, {funcHandle});

  // --- Step 7: Return the result slot pointer (caller extracts needed field)
  // ---

  return resultSlot;
}

} // namespace libtriton::conversion::utils
