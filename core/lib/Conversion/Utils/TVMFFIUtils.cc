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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

namespace trident::conversion::utils {

namespace {

constexpr int32_t kGlobalCtorDtorPriority = 65535;

mlir::FailureOr<mlir::LLVM::GlobalOp>
getOrCreateTVMFFIGlobalHandle(mlir::ModuleOp moduleOp,
                              llvm::StringRef funcName) {
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  std::string globalName =
      llvm::formatv("__trident_tvm_ffi_handle_{0}", funcName);

  if (mlir::LLVM::GlobalOp existing =
          moduleOp.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
    if (existing.getGlobalType() != ptrTy) {
      existing.emitError("cached TVM FFI handle has incompatible type");
      return mlir::failure();
    }
    return existing;
  }

  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToStart(moduleOp.getBody());
  return mlir::LLVM::GlobalOp::create(
      builder, moduleOp.getLoc(), ptrTy, /*isConstant=*/false,
      mlir::LLVM::Linkage::Internal, globalName,
      mlir::LLVM::ZeroAttr::get(ctx));
}

mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateTVMFFIGlobalCtor(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::ModuleOp moduleOp,
                            llvm::StringRef funcName,
                            mlir::LLVM::GlobalOp handleStorage) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Type i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::Type byteArrayTy =
      mlir::LLVM::LLVMStructType::getLiteral(ctx, {ptrTy, i64Ty});
  mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
  mlir::LLVM::LLVMFunctionType functionTy =
      mlir::LLVM::LLVMFunctionType::get(voidTy, {});
  std::string ctorName =
      llvm::formatv("__trident_tvm_ffi_ctor_{0}", funcName);

  if (mlir::LLVM::LLVMFuncOp existing =
          moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(ctorName)) {
    if (existing.getFunctionType() != functionTy) {
      existing.emitError("cached TVM FFI constructor has incompatible type");
      return mlir::failure();
    }
    return existing;
  }

  mlir::LLVM::LLVMFuncOp getGlobalCAPI = TRIDENT_CHECK_FAILURE(
      getOrCreateTVMFFIFunctionGetGlobal(moduleOp));

  builder.setInsertionPointToEnd(moduleOp.getBody());
  mlir::LLVM::LLVMFuncOp ctor = mlir::LLVM::LLVMFuncOp::create(
      builder, loc, ctorName, functionTy, mlir::LLVM::Linkage::Internal);
  mlir::Block *entryBlock = ctor.addEntryBlock(builder);
  builder.setInsertionPointToStart(entryBlock);

  mlir::Value namePtr =
      getOrCreateGlobalString(builder, loc, moduleOp, funcName, funcName);
  mlir::Value one = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1);
  mlir::Value nameSlot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, byteArrayTy, one);
  mlir::LLVM::StoreOp::create(
      builder, loc, namePtr,
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, byteArrayTy, nameSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
  mlir::LLVM::StoreOp::create(
      builder, loc,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, funcName.size()),
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, byteArrayTy, nameSlot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));

  mlir::Value funcSlot =
      mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, ptrTy, one);
  mlir::LLVM::CallOp::create(builder, loc, getGlobalCAPI,
                             {nameSlot, funcSlot});
  mlir::Value funcHandle =
      mlir::LLVM::LoadOp::create(builder, loc, ptrTy, funcSlot);
  mlir::Value globalAddress =
      mlir::LLVM::AddressOfOp::create(builder, loc, handleStorage).getResult();
  mlir::LLVM::StoreOp::create(builder, loc, funcHandle, globalAddress);
  mlir::LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
  return ctor;
}

mlir::FailureOr<mlir::LLVM::LLVMFuncOp>
getOrCreateTVMFFIGlobalDtor(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::ModuleOp moduleOp,
                            llvm::StringRef funcName,
                            mlir::LLVM::GlobalOp handleStorage) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Type voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
  mlir::LLVM::LLVMFunctionType functionTy =
      mlir::LLVM::LLVMFunctionType::get(voidTy, {});
  std::string dtorName =
      llvm::formatv("__trident_tvm_ffi_dtor_{0}", funcName);

  if (mlir::LLVM::LLVMFuncOp existing =
          moduleOp.lookupSymbol<mlir::LLVM::LLVMFuncOp>(dtorName)) {
    if (existing.getFunctionType() != functionTy) {
      existing.emitError("cached TVM FFI destructor has incompatible type");
      return mlir::failure();
    }
    return existing;
  }

  mlir::LLVM::LLVMFuncOp decRef =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIObjectDecRef(moduleOp));

  builder.setInsertionPointToEnd(moduleOp.getBody());
  mlir::LLVM::LLVMFuncOp dtor = mlir::LLVM::LLVMFuncOp::create(
      builder, loc, dtorName, functionTy, mlir::LLVM::Linkage::Internal);
  mlir::Block *entryBlock = dtor.addEntryBlock(builder);
  builder.setInsertionPointToStart(entryBlock);
  mlir::Value globalAddress =
      mlir::LLVM::AddressOfOp::create(builder, loc, handleStorage).getResult();
  mlir::Value funcHandle =
      mlir::LLVM::LoadOp::create(builder, loc, ptrTy, globalAddress);
  mlir::LLVM::CallOp::create(builder, loc, decRef, {funcHandle});
  mlir::LLVM::ReturnOp::create(builder, loc, mlir::ValueRange{});
  return dtor;
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
  auto ops = moduleOp.getOps<mlir::LLVM::GlobalCtorsOp>();
  if (ops.begin() != ops.end()) {
    mlir::LLVM::GlobalCtorsOp op = *ops.begin();
    if (op.getCtors().size() != op.getPriorities().size() ||
        op.getCtors().size() != op.getData().size()) {
      op.emitError("malformed global constructor registration");
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Attribute> ctors(op.getCtors().begin(),
                                              op.getCtors().end());
    llvm::SmallVector<mlir::Attribute> priorities(op.getPriorities().begin(),
                                                   op.getPriorities().end());
    llvm::SmallVector<mlir::Attribute> dataEntries(op.getData().begin(),
                                                   op.getData().end());
    ctors.push_back(ctorRef);
    priorities.push_back(priority);
    dataEntries.push_back(data);
    op.setCtorsAttr(builder.getArrayAttr(ctors));
    op.setPrioritiesAttr(builder.getArrayAttr(priorities));
    op.setDataAttr(builder.getArrayAttr(dataEntries));
    return mlir::success();
  }

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
  auto ops = moduleOp.getOps<mlir::LLVM::GlobalDtorsOp>();
  if (ops.begin() != ops.end()) {
    mlir::LLVM::GlobalDtorsOp op = *ops.begin();
    if (op.getDtors().size() != op.getPriorities().size() ||
        op.getDtors().size() != op.getData().size()) {
      op.emitError("malformed global destructor registration");
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Attribute> dtors(op.getDtors().begin(),
                                              op.getDtors().end());
    llvm::SmallVector<mlir::Attribute> priorities(op.getPriorities().begin(),
                                                   op.getPriorities().end());
    llvm::SmallVector<mlir::Attribute> dataEntries(op.getData().begin(),
                                                   op.getData().end());
    dtors.push_back(dtorRef);
    priorities.push_back(priority);
    dataEntries.push_back(data);
    op.setDtorsAttr(builder.getArrayAttr(dtors));
    op.setPrioritiesAttr(builder.getArrayAttr(priorities));
    op.setDataAttr(builder.getArrayAttr(dataEntries));
    return mlir::success();
  }

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

mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         mlir::Value argsArray, mlir::Value numArgs) {
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
        getOrCreateTVMFFIGlobalHandle(moduleOp, funcName));
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

  // Call TVMFFIFunctionCall with runtime numArgs and the cached handle.

  mlir::LLVM::LLVMFuncOp ffiCall =
      TRIDENT_CHECK_FAILURE(getOrCreateTVMFFIFunctionCall(moduleOp));
  mlir::LLVM::CallOp::create(builder, loc, ffiCall,
                             {funcHandle, argsArray, numArgs, resultSlot});

  return resultSlot;
}

} // namespace trident::conversion::utils
