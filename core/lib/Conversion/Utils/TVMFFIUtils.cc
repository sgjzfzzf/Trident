//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Conversion/Utils/GlobalString.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <tvm/ffi/c_api.h>

namespace trident::conversion::utils {

mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         llvm::ArrayRef<mlir::Value> args) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType const anyTy =
      tvm_ffi::TVMFFIABIType::getLLVMType(ctx);
  const size_t numArgsCount = args.size();

  // Allocate contiguous args array and copy each pre-built slot.
  mlir::Value const argsArray = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty,
                                     static_cast<int64_t>(numArgsCount)));
  for (auto [i, arg] : llvm::enumerate(args)) {
    mlir::Value const dst = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, anyTy, argsArray,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
    mlir::Value const loaded =
        mlir::LLVM::LoadOp::create(builder, loc, anyTy, arg);
    mlir::LLVM::StoreOp::create(builder, loc, loaded, dst);
  }

  mlir::Value const numArgs = mlir::LLVM::ConstantOp::create(
      builder, loc, i32Ty, static_cast<int64_t>(numArgsCount));
  return callTVMFFIGlobalFunction(builder, loc, moduleOp, funcName, argsArray,
                                  numArgs);
}

mlir::FailureOr<mlir::Value> getTVMFFIGlobalFunction(mlir::OpBuilder &builder,
                                                     mlir::Location loc,
                                                     mlir::ModuleOp moduleOp,
                                                     llvm::StringRef funcName) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType const anyTy =
      tvm_ffi::TVMFFIABIType::getLLVMType(ctx);

  mlir::LLVM::LLVMStructType const byteArrayTy =
      mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  mlir::Value const namePtr =
      getOrCreateGlobalString(builder, loc, moduleOp, funcName, funcName);
  mlir::Value const nameSlot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, byteArrayTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::LLVM::StoreOp::create(
      builder, loc, namePtr,
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, byteArrayTy, nameSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
  mlir::LLVM::StoreOp::create(
      builder, loc,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty,
                                     static_cast<int64_t>(funcName.size())),
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, byteArrayTy, nameSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));

  mlir::Value const funcSlot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, ptrTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getGlobal =
      getOrCreateTVMFFIFunctionGetGlobal(moduleOp);
  if (mlir::failed(getGlobal))
    return mlir::failure();
  mlir::LLVM::CallOp::create(builder, loc, getGlobal.value(),
                             {nameSlot, funcSlot});
  return mlir::LLVM::LoadOp::create(builder, loc, ptrTy, funcSlot).getResult();
}

mlir::LogicalResult
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   llvm::ArrayRef<mlir::Value> args, mlir::Value resultSlot) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType const anyTy =
      tvm_ffi::TVMFFIABIType::getLLVMType(ctx);
  mlir::Value const argsArray = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, anyTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty,
                                     static_cast<int64_t>(args.size())));
  for (auto [index, arg] : llvm::enumerate(args)) {
    mlir::Value const dst = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, anyTy, argsArray,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
    mlir::Value const load =
        mlir::LLVM::LoadOp::create(builder, loc, anyTy, arg);
    mlir::LLVM::StoreOp::create(builder, loc, load, dst);
  }
  mlir::Value const numArgs = mlir::LLVM::ConstantOp::create(
      builder, loc, i32Ty, static_cast<int32_t>(args.size()));
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
  mlir::LLVM::CallOp::create(builder, loc, ffiCall.value(),
                             {funcHandle, argsArray, numArgs, resultSlot});

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> decRef =
      getOrCreateTVMFFIObjectDecRef(moduleOp);
  if (mlir::failed(decRef)) {
    return mlir::failure();
  }
  mlir::LLVM::CallOp::create(builder, loc, decRef.value(), {funcHandle});
  return mlir::success();
}

mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         mlir::Value argsArray, mlir::Value numArgs) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::LLVM::LLVMStructType const anyTy =
      tvm_ffi::TVMFFIABIType::getLLVMType(ctx);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::FailureOr<mlir::Value> funcHandle =
      getTVMFFIGlobalFunction(builder, loc, moduleOp, funcName);
  if (mlir::failed(funcHandle))
    return mlir::failure();

  mlir::Value const zero32 =
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, 0);
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
  if (mlir::failed(callTVMFFIFunction(builder, loc, moduleOp,
                                      funcHandle.value(), argsArray, numArgs,
                                      resultSlot))) {
    return mlir::failure();
  }
  return resultSlot;
}

} // namespace trident::conversion::utils
