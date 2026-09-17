//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/String.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <tuple>

namespace trident::conversion::utils {

namespace {

/// Get or create the module-level global caching the handle of \p funcName.
mlir::FailureOr<mlir::LLVM::GlobalOp>
getOrCreateTVMFFIGlobalHandle(mlir::ModuleOp moduleOp,
                              llvm::StringRef funcName) {
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type const ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  std::string const globalName =
      llvm::formatv("__trident_tvm_ffi_handle_{0}", funcName);

  if (mlir::LLVM::GlobalOp existing =
          moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
    if (existing.getGlobalType() != ptrTy) {
      return existing.emitError("cached TVM FFI handle has incompatible type");
    }
    return existing;
  }

  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(moduleOp.getBody());
  return mlir::LLVM::GlobalOp::create(builder, moduleOp.getLoc(), ptrTy,
                                      /*isConstant=*/false,
                                      mlir::LLVM::Linkage::Internal, globalName,
                                      mlir::LLVM::ZeroAttr::get(ctx));
}

/// Get or create a `void ()` module-level function whose body is emitted by
/// \p buildBody. \p description names the function in diagnostics.
mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getOrCreateTVMFFIHandleLifecycle(
    mlir::ModuleOp moduleOp, llvm::StringRef prefix, llvm::StringRef funcName,
    llvm::StringRef description,
    llvm::function_ref<mlir::LogicalResult(mlir::OpBuilder &, mlir::Location)>
        buildBody) {
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Location const loc = moduleOp.getLoc();
  mlir::LLVM::LLVMFunctionType const functionType =
      mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(ctx), {});
  std::string const symbolName = llvm::formatv("{0}{1}", prefix, funcName);

  if (mlir::LLVM::LLVMFuncOp existing =
          moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(symbolName)) {
    if (existing.getFunctionType() != functionType) {
      return existing.emitError() << "cached TVM FFI handle " << description
                                  << " has incompatible type";
    }
    return existing;
  }

  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(moduleOp.getBody());
  mlir::LLVM::LLVMFuncOp function = mlir::LLVM::LLVMFuncOp::create(
      builder, loc, symbolName, functionType, mlir::LLVM::Linkage::Internal);
  builder.setInsertionPointToStart(function.addEntryBlock(builder));
  if (mlir::failed(buildBody(builder, loc))) {
    function.erase();
    return mlir::failure();
  }
  mlir::LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
  return function;
}

/// Resolve \p funcName once at load time and cache the handle and the lookup
/// status in module-level storage.
mlir::LogicalResult
buildTVMFFIHandleCtorBody(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                          mlir::LLVM::GlobalOp handleGlobal) {
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType const byteArrayTy =
      mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

  mlir::LLVM::LLVMFuncOp getGlobal =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIFunctionGetGlobal(moduleOp));

  mlir::Value const namePtr = getString(builder, loc, funcName);
  mlir::Value const one =
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1);
  mlir::Value const nameSlot =
      mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, byteArrayTy, one);
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

  mlir::Value const funcSlot =
      mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, ptrTy, one);
  mlir::LLVM::CallOp::create(builder, loc, getGlobal, {nameSlot, funcSlot});
  mlir::LLVM::StoreOp::create(
      builder, loc, mlir::LLVM::LoadOp::create(builder, loc, ptrTy, funcSlot),
      mlir::LLVM::AddressOfOp::create(builder, loc, handleGlobal));
  return mlir::success();
}

/// Release the cached handle at unload time.
mlir::LogicalResult
buildTVMFFIHandleDtorBody(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::ModuleOp moduleOp,
                          mlir::LLVM::GlobalOp handleGlobal) {
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(moduleOp.getContext());

  mlir::LLVM::LLVMFuncOp decRef =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIObjectDecRef(moduleOp));
  mlir::Value const handle = mlir::LLVM::LoadOp::create(
      builder, loc, ptrTy,
      mlir::LLVM::AddressOfOp::create(builder, loc, handleGlobal));
  mlir::LLVM::CallOp::create(builder, loc, decRef, {handle});
  return mlir::success();
}

/// Add \p ctor to the module's `llvm.mlir.global_ctors`, unless already listed.
void registerGlobalCtor(mlir::ModuleOp moduleOp, mlir::LLVM::LLVMFuncOp ctor) {
  for (mlir::LLVM::GlobalCtorsOp list :
       moduleOp.getOps<mlir::LLVM::GlobalCtorsOp>()) {
    if (llvm::any_of(list.getCtors(), [&](mlir::Attribute attr) {
          auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
          return symbol && symbol.getValue() == ctor.getSymName();
        })) {
      return;
    }
  }

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToEnd(moduleOp.getBody());
  // Constructors run in ascending priority order, so the maximum resolves
  // handles last on load.
  mlir::LLVM::GlobalCtorsOp::create(
      builder, moduleOp.getLoc(),
      builder.getArrayAttr({mlir::FlatSymbolRefAttr::get(moduleOp.getContext(),
                                                         ctor.getSymName())}),
      builder.getArrayAttr({builder.getI32IntegerAttr(65535)}),
      builder.getArrayAttr({mlir::LLVM::ZeroAttr::get(moduleOp.getContext())}));
}

/// Add \p dtor to the module's `llvm.mlir.global_dtors`, unless already listed.
void registerGlobalDtor(mlir::ModuleOp moduleOp, mlir::LLVM::LLVMFuncOp dtor) {
  for (mlir::LLVM::GlobalDtorsOp list :
       moduleOp.getOps<mlir::LLVM::GlobalDtorsOp>()) {
    if (llvm::any_of(list.getDtors(), [&](mlir::Attribute attr) {
          auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
          return symbol && symbol.getValue() == dtor.getSymName();
        })) {
      return;
    }
  }

  mlir::OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToEnd(moduleOp.getBody());
  // Destructors run in descending priority order, so the maximum releases
  // handles first on unload.
  mlir::LLVM::GlobalDtorsOp::create(
      builder, moduleOp.getLoc(),
      builder.getArrayAttr({mlir::FlatSymbolRefAttr::get(moduleOp.getContext(),
                                                         dtor.getSymName())}),
      builder.getArrayAttr({builder.getI32IntegerAttr(65535)}),
      builder.getArrayAttr({mlir::LLVM::ZeroAttr::get(moduleOp.getContext())}));
}

/// Resolve \p funcName into the module-level cache and load the handle it
/// holds. The module owns that reference for its whole lifetime, so the
/// returned handle is borrowed.
mlir::FailureOr<mlir::Value>
loadCachedTVMFFIGlobalHandle(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::ModuleOp moduleOp,
                             llvm::StringRef funcName) {
  mlir::LLVM::GlobalOp handleGlobal =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIGlobalHandle(moduleOp, funcName));

  mlir::LLVM::LLVMFuncOp ctor =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIHandleLifecycle(
          moduleOp, "__trident_tvm_ffi_ctor_", funcName, "constructor",
          [&](mlir::OpBuilder &bodyBuilder,
              mlir::Location bodyLoc) -> mlir::LogicalResult {
            return buildTVMFFIHandleCtorBody(bodyBuilder, bodyLoc, moduleOp,
                                             funcName, handleGlobal);
          }));
  mlir::LLVM::LLVMFuncOp dtor =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIHandleLifecycle(
          moduleOp, "__trident_tvm_ffi_dtor_", funcName, "destructor",
          [&](mlir::OpBuilder &bodyBuilder,
              mlir::Location bodyLoc) -> mlir::LogicalResult {
            return buildTVMFFIHandleDtorBody(bodyBuilder, bodyLoc, moduleOp,
                                             handleGlobal);
          }));
  registerGlobalCtor(moduleOp, ctor);
  registerGlobalDtor(moduleOp, dtor);

  return mlir::LLVM::LoadOp::create(
             builder, loc,
             mlir::LLVM::LLVMPointerType::get(moduleOp.getContext()),
             mlir::LLVM::AddressOfOp::create(builder, loc, handleGlobal))
      .getResult();
}

} // namespace

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

  // This helper emits LLVM directly, so the ownership deallocation pass never
  // sees it. The module holds the cached reference for its whole lifetime and
  // the call cannot outlive it, so borrowing the handle is enough.
  mlir::Value const handle = TRIDENT_CHECK_FAILURE(
      loadCachedTVMFFIGlobalHandle(builder, loc, moduleOp, funcName));

  mlir::Value const zero32 =
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, 0);
  mlir::Value const resultSlot = mlir::LLVM::AllocaOp::create(
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

  if (mlir::failed(callTVMFFIFunction(builder, loc, moduleOp, handle, argsArray,
                                      numArgs, resultSlot))) {
    return mlir::failure();
  }
  return resultSlot;
}

mlir::FailureOr<std::tuple<mlir::Value, mlir::Value>>
getTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::ModuleOp moduleOp, llvm::StringRef funcName) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType const i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(ctx);

  mlir::Value const handle = TRIDENT_CHECK_FAILURE(
      loadCachedTVMFFIGlobalHandle(builder, loc, moduleOp, funcName));
  // The cached reference belongs to the module, so retain one for the caller.
  // `FunctionGetGlobalOp` keeps its default `Owned` result contract, and the
  // ownership deallocation pass pairs this retain with a release.
  mlir::LLVM::LLVMFuncOp incRef =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIObjectIncRef(moduleOp));
  mlir::LLVM::CallOp::create(builder, loc, incRef, {handle});
  // The constructor leaves the global null when the lookup failed, so the
  // status the caller checks is derived from the cached handle: zero, i.e.
  // success, exactly when a handle was resolved.
  mlir::Value const status = mlir::LLVM::ZExtOp::create(
      builder, loc, i32Ty,
      mlir::LLVM::ICmpOp::create(
          builder, loc, mlir::LLVM::ICmpPredicate::eq, handle,
          mlir::LLVM::ZeroOp::create(builder, loc, ptrTy)));
  return std::make_tuple(handle, status);
}

mlir::FailureOr<mlir::Value>
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

mlir::FailureOr<mlir::Value>
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   mlir::Value argsArray, mlir::Value numArgs,
                   mlir::Value resultSlot) {
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> ffiCall =
      getOrCreateTVMFFIFunctionCall(moduleOp);
  if (mlir::failed(ffiCall)) {
    return mlir::failure();
  }
  mlir::LLVM::CallOp call =
      mlir::LLVM::CallOp::create(builder, loc, ffiCall.value(),
                                 {funcHandle, argsArray, numArgs, resultSlot});
  return call.getResult();
}
} // namespace trident::conversion::utils
