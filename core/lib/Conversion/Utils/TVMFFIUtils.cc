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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace trident::conversion::utils {

namespace {

constexpr int32_t kGlobalCtorDtorPriority = 65535;

mlir::FailureOr<mlir::LLVM::GlobalOp>
getOrCreateTVMFFIGlobalHandle(mlir::OpBuilder &builder,
                              mlir::ModuleOp moduleOp,
                              llvm::StringRef funcName) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  std::string globalName =
      llvm::formatv("__trident_tvm_ffi_handle_{0}", funcName);

  if (mlir::LLVM::GlobalOp existing =
          moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
    if (existing.getGlobalType() != ptrTy)
      return existing.emitError(
          "cached TVM FFI handle has incompatible type");
    return existing;
  }

  builder.setInsertionPointToStart(moduleOp.getBody());
  return mlir::LLVM::GlobalOp::create(
      builder, moduleOp.getLoc(), ptrTy, /*isConstant=*/false,
      mlir::LLVM::Linkage::Internal, globalName,
      mlir::LLVM::ZeroAttr::get(ctx));
}

mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateTVMFFIGlobalLifecycle(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::ModuleOp moduleOp,
    llvm::StringRef funcName, llvm::StringRef symbolPrefix,
    llvm::StringRef description,
    llvm::function_ref<mlir::LogicalResult()> buildBody) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
  mlir::LLVM::LLVMFunctionType functionTy =
      mlir::LLVM::LLVMFunctionType::get(voidTy, {});
  std::string symbolName = llvm::formatv("{0}{1}", symbolPrefix, funcName);

  if (mlir::LLVM::LLVMFuncOp existing =
          moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(symbolName)) {
    if (existing.getFunctionType() != functionTy)
      return existing.emitError()
             << "cached TVM FFI " << description
             << " has incompatible type";
    return existing;
  }

  builder.setInsertionPointToEnd(moduleOp.getBody());
  mlir::LLVM::LLVMFuncOp function = mlir::LLVM::LLVMFuncOp::create(
      builder, loc, symbolName, functionTy, mlir::LLVM::Linkage::Internal);
  mlir::Block *entryBlock = function.addEntryBlock(builder);
  builder.setInsertionPointToStart(entryBlock);

  if (mlir::failed(buildBody())) {
    function.erase();
    return mlir::emitError(loc)
           << "failed to build TVM FFI global " << description;
  }

  mlir::LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
  return function;
}

mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateTVMFFIGlobalCtor(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::ModuleOp moduleOp,
                            llvm::StringRef funcName,
                            mlir::LLVM::GlobalOp handleStorage) {
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Type i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::Type byteArrayTy =
      mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});

  return getOrCreateTVMFFIGlobalLifecycle(
      builder, loc, moduleOp, funcName, "__trident_tvm_ffi_ctor_",
      "constructor", [&]() -> mlir::LogicalResult {
        mlir::LLVM::LLVMFuncOp getGlobalCAPI = TRIDENT_CHECK_FAILURE(
            getOrCreateTVMFFIFunctionGetGlobal(moduleOp));
        mlir::Value namePtr =
            getOrCreateGlobalString(builder, loc, moduleOp, funcName, funcName);
        mlir::Value one =
            mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1);
        mlir::Value nameSlot = mlir::LLVM::AllocaOp::create(
            builder, loc, ptrTy, byteArrayTy, one);
        mlir::LLVM::StoreOp::create(
            builder, loc, namePtr,
            mlir::LLVM::GEPOp::create(
                builder, loc, ptrTy, byteArrayTy, nameSlot,
                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
        mlir::LLVM::StoreOp::create(
            builder, loc,
            mlir::LLVM::ConstantOp::create(builder, loc, i64Ty,
                                            funcName.size()),
            mlir::LLVM::GEPOp::create(
                builder, loc, ptrTy, byteArrayTy, nameSlot,
                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));

        mlir::Value funcSlot =
            mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, ptrTy, one);
        mlir::LLVM::CallOp::create(builder, loc, getGlobalCAPI,
                                   {nameSlot, funcSlot});
        mlir::Value funcHandle =
            mlir::LLVM::LoadOp::create(builder, loc, ptrTy, funcSlot);
        mlir::Value globalAddress = mlir::LLVM::AddressOfOp::create(
            builder, loc, handleStorage);
        mlir::LLVM::StoreOp::create(builder, loc, funcHandle, globalAddress);
        return mlir::success();
      });
}

mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateTVMFFIGlobalDtor(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::ModuleOp moduleOp,
                            llvm::StringRef funcName,
                            mlir::LLVM::GlobalOp handleStorage) {
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(moduleOp.getContext());

  return getOrCreateTVMFFIGlobalLifecycle(
      builder, loc, moduleOp, funcName, "__trident_tvm_ffi_dtor_",
      "destructor", [&]() -> mlir::LogicalResult {
        mlir::LLVM::LLVMFuncOp decRef =
            TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIObjectDecRef(moduleOp));
        mlir::Value globalAddress = mlir::LLVM::AddressOfOp::create(
            builder, loc, handleStorage);
        mlir::Value funcHandle =
            mlir::LLVM::LoadOp::create(builder, loc, ptrTy, globalAddress);
        mlir::LLVM::CallOp::create(builder, loc, decRef, {funcHandle});
        return mlir::success();
      });
}

mlir::LogicalResult registerGlobalCtor(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::ModuleOp moduleOp,
                                       mlir::LLVM::LLVMFuncOp ctor) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  for (mlir::LLVM::GlobalCtorsOp op :
       moduleOp.getOps<mlir::LLVM::GlobalCtorsOp>()) {
    if (llvm::any_of(op.getCtors(), [&](mlir::Attribute attr) {
          auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
          return symbol && symbol.getValue() == ctor.getSymName();
        }))
      return mlir::success();
  }

  mlir::Attribute ctorRef =
      mlir::FlatSymbolRefAttr::get(moduleOp.getContext(), ctor.getSymName());
  mlir::Attribute priority = builder.getI32IntegerAttr(kGlobalCtorDtorPriority);
  mlir::Attribute data = mlir::LLVM::ZeroAttr::get(moduleOp.getContext());

  builder.setInsertionPointToEnd(moduleOp.getBody());
  mlir::LLVM::GlobalCtorsOp::create(
      builder, loc, builder.getArrayAttr({ctorRef}),
      builder.getArrayAttr({priority}), builder.getArrayAttr({data}));
  return mlir::success();
}

mlir::LogicalResult registerGlobalDtor(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::ModuleOp moduleOp,
                                       mlir::LLVM::LLVMFuncOp dtor) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  for (mlir::LLVM::GlobalDtorsOp op :
       moduleOp.getOps<mlir::LLVM::GlobalDtorsOp>()) {
    if (llvm::any_of(op.getDtors(), [&](mlir::Attribute attr) {
          auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
          return symbol && symbol.getValue() == dtor.getSymName();
        }))
      return mlir::success();
  }

  mlir::Attribute dtorRef =
      mlir::FlatSymbolRefAttr::get(moduleOp.getContext(), dtor.getSymName());
  mlir::Attribute priority = builder.getI32IntegerAttr(kGlobalCtorDtorPriority);
  mlir::Attribute data = mlir::LLVM::ZeroAttr::get(moduleOp.getContext());

  builder.setInsertionPointToEnd(moduleOp.getBody());
  mlir::LLVM::GlobalDtorsOp::create(
      builder, loc, builder.getArrayAttr({dtorRef}),
      builder.getArrayAttr({priority}), builder.getArrayAttr({data}));
  return mlir::success();
}

} // namespace

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

  mlir::LLVM::GlobalOp handleGlobal;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    handleGlobal = TRIDENT_CHECK_FAILURE(
        getOrCreateTVMFFIGlobalHandle(builder, moduleOp, funcName));
    mlir::LLVM::LLVMFuncOp ctor = TRIDENT_CHECK_FAILURE(
        getOrCreateTVMFFIGlobalCtor(builder, loc, moduleOp, funcName,
                                    handleGlobal));
    mlir::LLVM::LLVMFuncOp dtor = TRIDENT_CHECK_FAILURE(
        getOrCreateTVMFFIGlobalDtor(builder, loc, moduleOp, funcName,
                                    handleGlobal));
    if (mlir::failed(registerGlobalCtor(builder, loc, moduleOp, ctor)) ||
        mlir::failed(registerGlobalDtor(builder, loc, moduleOp, dtor)))
      return mlir::failure();
  }
  mlir::Value globalAddress =
      mlir::LLVM::AddressOfOp::create(builder, loc, handleGlobal).getResult();
  mlir::Value funcHandle =
      mlir::LLVM::LoadOp::create(builder, loc, ptrTy, globalAddress);

  // Allocate and zero-initialize result slot.

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

  // Call TVMFFIFunctionCall with runtime numArgs and the cached handle.

  mlir::LLVM::LLVMFuncOp ffiCall =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIFunctionCall(moduleOp));
  mlir::LLVM::CallOp::create(builder, loc, ffiCall,
                             {funcHandle, argsArray, numArgs, resultSlot});

  return resultSlot;
}

} // namespace trident::conversion::utils
