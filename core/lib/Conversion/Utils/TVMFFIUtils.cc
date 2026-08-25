//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "trident/core/Conversion/Utils/GlobalString.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "tvm/ffi/c_api.h"

namespace trident::conversion::utils {

mlir::Value buildIntAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                            int64_t value) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  mlir::Value slot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::Value any = mlir::LLVM::UndefOp::create(builder, loc, anyTy);
  any = mlir::LLVM::InsertValueOp::create(
      builder, loc, any,
      mlir::LLVM::ConstantOp::create(
          builder, loc, i32Ty, ::trident::tvm_ffi::IntType::getTypeIndex()),
      llvm::ArrayRef<int64_t>{0});
  any = mlir::LLVM::InsertValueOp::create(
      builder, loc, any, mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, 0),
      llvm::ArrayRef<int64_t>{1});
  any = mlir::LLVM::InsertValueOp::create(
      builder, loc, any,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, value),
      llvm::ArrayRef<int64_t>{2});
  mlir::LLVM::StoreOp::create(builder, loc, any, slot);
  return slot;
}

mlir::Value buildIntAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Value slot = buildIntAnySlot(builder, loc, 0);
  mlir::LLVM::StoreOp::create(
      builder, loc, mlir::LLVM::SExtOp::create(builder, loc, i64Ty, value),
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
  return slot;
}

mlir::Value buildDTypeAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                              int64_t code, int64_t bits, int64_t lanes) {
  const int64_t payload =
      (code & 0xff) | ((bits & 0xff) << 8) | ((lanes & 0xffff) << 16);
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::Value slot = buildIntAnySlot(builder, loc, payload);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::Value typeIndex = mlir::LLVM::ConstantOp::create(
      builder, loc, i32Ty, ::trident::tvm_ffi::DTypeType::getTypeIndex());
  mlir::LLVM::StoreOp::create(
      builder, loc, typeIndex,
      mlir::LLVM::GEPOp::create(
          builder, loc, mlir::LLVM::LLVMPointerType::get(ctx), anyTy, slot,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
  return slot;
}

mlir::Value buildOpaquePtrAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value pointer) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::Value slot = buildIntAnySlot(builder, loc, 0);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Value typeIndex =
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, kTVMFFIOpaquePtr);
  mlir::LLVM::StoreOp::create(
      builder, loc, typeIndex,
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
  mlir::Value payload =
      mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, pointer);
  mlir::LLVM::StoreOp::create(
      builder, loc, payload,
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
  return slot;
}

mlir::Value loadIntFromAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value slot) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Value payloadPtr =
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
  mlir::Value payload =
      mlir::LLVM::LoadOp::create(builder, loc, i64Ty, payloadPtr);
  return mlir::LLVM::TruncOp::create(builder, loc, i32Ty, payload);
}

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
  const size_t numArgsCount = args.size();

  // Allocate contiguous args array and copy each pre-built slot.
  mlir::Value argsArray = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, numArgsCount));
  for (auto [i, arg] : llvm::enumerate(args)) {
    mlir::Value dst =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, argsArray,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(builder, loc, anyTy, arg);
    mlir::LLVM::StoreOp::create(builder, loc, loaded, dst);
  }

  mlir::Value numArgs =
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, numArgsCount);
  return callTVMFFIGlobalFunction(builder, loc, moduleOp, funcName, argsArray,
                                  numArgs);
}

mlir::FailureOr<mlir::Value> getTVMFFIGlobalFunction(mlir::OpBuilder &builder,
                                                     mlir::Location loc,
                                                     mlir::ModuleOp moduleOp,
                                                     llvm::StringRef funcName) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy =
      trident::conversion::utils::getTVMFFIAnyType(ctx);

  mlir::LLVM::LLVMStructType byteArrayTy =
      mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
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

  mlir::Value funcSlot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, ptrTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getGlobal =
      getOrCreateTVMFFIFunctionGetGlobal(moduleOp);
  if (mlir::failed(getGlobal))
    return mlir::failure();
  mlir::LLVM::CallOp::create(builder, loc, *getGlobal, {nameSlot, funcSlot});
  return mlir::LLVM::LoadOp::create(builder, loc, ptrTy, funcSlot).getResult();
}

mlir::LogicalResult
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   llvm::ArrayRef<mlir::Value> args, mlir::Value resultSlot) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  mlir::Value argsArray = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, args.size()));
  for (auto [index, arg] : llvm::enumerate(args)) {
    mlir::Value dst =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, argsArray,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
    mlir::Value load = mlir::LLVM::LoadOp::create(builder, loc, anyTy, arg);
    mlir::LLVM::StoreOp::create(builder, loc, load, dst);
  }
  mlir::Value numArgs =
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, args.size());
  return callTVMFFIFunction(builder, loc, moduleOp, funcHandle, argsArray,
                            numArgs, resultSlot);
}

mlir::LogicalResult
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   mlir::Value argsArray, mlir::Value numArgs,
                   mlir::Value resultSlot) {
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> ffiCall =
      getOrCreateTVMFFIFunctionCall(moduleOp);
  if (mlir::failed(ffiCall)) {
    return mlir::failure();
  }
  mlir::LLVM::CallOp::create(builder, loc, *ffiCall,
                             {funcHandle, argsArray, numArgs, resultSlot});

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> decRef =
      getOrCreateTVMFFIObjectDecRef(moduleOp);
  if (mlir::failed(decRef)) {
    return mlir::failure();
  }
  mlir::LLVM::CallOp::create(builder, loc, *decRef, {funcHandle});
  return mlir::success();
}

mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         mlir::Value argsArray, mlir::Value numArgs) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::FailureOr<mlir::Value> funcHandle =
      getTVMFFIGlobalFunction(builder, loc, moduleOp, funcName);
  if (mlir::failed(funcHandle))
    return mlir::failure();

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
  mlir::LLVM::StoreOp::create(
      builder, loc, mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0),
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, resultSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
  if (mlir::failed(callTVMFFIFunction(builder, loc, moduleOp, *funcHandle,
                                      argsArray, numArgs, resultSlot))) {
    return mlir::failure();
  }
  return resultSlot;
}

} // namespace trident::conversion::utils
