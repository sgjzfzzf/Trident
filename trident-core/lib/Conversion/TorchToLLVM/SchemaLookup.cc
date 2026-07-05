//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "SchemaLookup.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/function_schema.h"
#include "ATen/core/jit_type_base.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "trident-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident-core/Conversion/Utils/CFunctionDeclUtils.h"
#include "trident-core/Conversion/Utils/GlobalString.h"
#include "trident-core/Conversion/Utils/StableCAPIDescriptors.h"
#include "trident-core/Conversion/Utils/StdLibCAPIDescriptors.h"
#include "trident-core/Conversion/Utils/TVMFFIUtils.h"
#include "trident-core/Conversion/Utils/TridentCAPIDescriptors.h"
#include "trident-core/Conversion/Utils/Type.h"
#include "trident-core/Conversion/Utils/Unwrap.h"
#include "tvm/ffi/c_api.h"
#include <regex>
#include <string>
#include <vector>

namespace {

//===----------------------------------------------------------------------===//
// TVMFFIAny helpers
//===----------------------------------------------------------------------===//

/// Build a TVMFFIAny struct value from a type_index and i64 payload.
static mlir::Value buildFFIAnyValue(mlir::OpBuilder &builder,
                                    mlir::Location loc, int32_t typeIndex,
                                    mlir::Value payload) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::LLVM::LLVMStructType anyTy =
      trident::conversion::utils::getTVMFFIAnyType(ctx);

  mlir::Value result = mlir::LLVM::UndefOp::create(builder, loc, anyTy);
  mlir::Value typeIdx =
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, typeIndex);
  result = mlir::LLVM::InsertValueOp::create(
      builder, loc, anyTy, result, typeIdx, llvm::ArrayRef<int64_t>{0});
  return mlir::LLVM::InsertValueOp::create(builder, loc, anyTy, result, payload,
                                           llvm::ArrayRef<int64_t>{2})
      .getResult();
}

//===----------------------------------------------------------------------===//
// ShimValue building from c10 type info
//===----------------------------------------------------------------------===//

static mlir::Type getShimElemType(mlir::MLIRContext *ctx,
                                  const c10::TypePtr &c10Type) {
  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType:
    return mlir::LLVM::LLVMPointerType::get(ctx);
  case c10::TypeKind::BoolType:
    return mlir::IntegerType::get(ctx, 32);
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType:
    return mlir::IntegerType::get(ctx, 64);
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType:
  case c10::TypeKind::NumberType:
    return mlir::Float64Type::get(ctx);
  default:
    return {};
  }
}

static mlir::FailureOr<mlir::Value>
buildShimTensorValue(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value adaptedValue, mlir::ModuleOp moduleOp) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

  mlir::Value handleInt = mlir::LLVM::ExtractValueOp::create(
      builder, loc, adaptedValue, llvm::ArrayRef<int64_t>{2});
  mlir::Value handle =
      mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, handleInt);

  mlir::LLVM::LLVMFuncOp unpackFn = TRIDENT_UNWRAP_FAILURE(
      trident::conversion::utils::getOrCreateTVMFFIObjectToTensor(moduleOp));

  mlir::Value slot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, ptrTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::LLVM::CallOp::create(builder, loc, unpackFn, {handle, slot});
  return mlir::LLVM::LoadOp::create(builder, loc, ptrTy, slot).getResult();
}

static mlir::Value buildShimNumberValue(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value adaptedValue) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::Float64Type f64Ty = mlir::Float64Type::get(ctx);

  mlir::Value typeIndex = mlir::LLVM::ExtractValueOp::create(
      builder, loc, adaptedValue, llvm::ArrayRef<int64_t>{0});
  mlir::Value payload = mlir::LLVM::ExtractValueOp::create(
      builder, loc, adaptedValue, llvm::ArrayRef<int64_t>{2});
  mlir::Value intTypeIndex =
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, kTVMFFIInt);
  mlir::Value isInt = mlir::LLVM::ICmpOp::create(
      builder, loc, mlir::LLVM::ICmpPredicate::eq, typeIndex, intTypeIndex);
  mlir::Value intAsDouble =
      mlir::LLVM::SIToFPOp::create(builder, loc, f64Ty, payload);
  mlir::Value floatAsDouble =
      mlir::LLVM::BitcastOp::create(builder, loc, f64Ty, payload);
  return mlir::LLVM::SelectOp::create(builder, loc, isInt, intAsDouble,
                                      floatAsDouble)
      .getResult();
}

static mlir::FailureOr<mlir::Value>
resolveShimTensorValue(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value shimValue, mlir::ModuleOp moduleOp) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

  mlir::LLVM::LLVMFuncOp packFn = TRIDENT_UNWRAP_FAILURE(
      trident::conversion::utils::getOrCreateTensorToTVMFFIObject(moduleOp));

  mlir::Value slot = mlir::LLVM::AllocaOp::create(
      builder, loc, ptrTy, ptrTy,
      mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
  mlir::LLVM::CallOp::create(builder, loc, packFn, {shimValue, slot});

  mlir::Value handle = mlir::LLVM::LoadOp::create(builder, loc, ptrTy, slot);
  mlir::Value payload =
      mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, handle);
  return buildFFIAnyValue(builder, loc, kTVMFFITensor, payload);
}

static mlir::FailureOr<mlir::Value> buildShimValue(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   const c10::TypePtr &c10Type,
                                                   mlir::Value adaptedValue,
                                                   mlir::ModuleOp moduleOp) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::Float64Type f64Ty = mlir::Float64Type::get(ctx);

  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType:
    return buildShimTensorValue(builder, loc, adaptedValue, moduleOp);
  case c10::TypeKind::BoolType:
    return mlir::LLVM::TruncOp::create(
               builder, loc, i32Ty,
               mlir::LLVM::ExtractValueOp::create(builder, loc, adaptedValue,
                                                  llvm::ArrayRef<int64_t>{2}))
        .getResult();
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType:
    return mlir::LLVM::ExtractValueOp::create(builder, loc, adaptedValue,
                                              llvm::ArrayRef<int64_t>{2})
        .getResult();
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType:
    return mlir::LLVM::BitcastOp::create(
               builder, loc, f64Ty,
               mlir::LLVM::ExtractValueOp::create(builder, loc, adaptedValue,
                                                  llvm::ArrayRef<int64_t>{2}))
        .getResult();
  case c10::TypeKind::NumberType:
    return buildShimNumberValue(builder, loc, adaptedValue);
  default:
    return mlir::failure();
  }
}

static mlir::FailureOr<mlir::Value>
resolveShimValue(mlir::OpBuilder &builder, mlir::Location loc,
                 const c10::TypePtr &c10Type, mlir::Value shimValue,
                 mlir::ModuleOp moduleOp) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);

  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType:
    return resolveShimTensorValue(builder, loc, shimValue, moduleOp);
  case c10::TypeKind::BoolType: {
    mlir::Value payload =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, shimValue);
    return buildFFIAnyValue(builder, loc, kTVMFFIBool, payload);
  }
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType:
    return buildFFIAnyValue(builder, loc, kTVMFFIInt, shimValue);
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType:
  case c10::TypeKind::NumberType: {
    mlir::Value payload =
        mlir::LLVM::BitcastOp::create(builder, loc, i64Ty, shimValue);
    return buildFFIAnyValue(builder, loc, kTVMFFIFloat, payload);
  }
  default:
    return mlir::failure();
  }
}

//===----------------------------------------------------------------------===//
// StableIValue building from c10 type info
//===----------------------------------------------------------------------===//

/// Converts an adapted (LLVM-typed) MLIR Value into a type-erased StableIValue
/// (uint64_t) based on the c10 type from the operator schema.
///
/// \param builder  The MLIR op builder.
/// \param c10Type  The c10 type from the op schema argument.
/// \param input    The adapted LLVM-typed MLIR value.
/// \param moduleOp The parent MLIR module.
/// \param loc      The source location.
/// \return The i64 StableIValue on success, or failure.
mlir::FailureOr<mlir::Value> buildStableIValue(mlir::OpBuilder &builder,
                                               const c10::TypePtr &c10Type,
                                               mlir::Value input,
                                               mlir::ModuleOp moduleOp,
                                               mlir::Location loc) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy =
      trident::conversion::utils::getTVMFFIAnyType(ctx);

  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType: {
    // Tensor: adapted value is a TVMFFIAny with payload = TVMFFIObjectHandle.
    // Extract the handle and unpack to AtenTensorHandle for the dispatcher.
    mlir::Value handleInt = mlir::LLVM::ExtractValueOp::create(
        builder, loc, input, llvm::ArrayRef<int64_t>{2});
    mlir::Value handle =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, handleInt);

    mlir::LLVM::LLVMFuncOp unpackFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateTVMFFIObjectToTensor(moduleOp));

    // Allocate stack slot for AtenTensorHandle output.
    mlir::Value slot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    // Call mTridentTVMFFIObjectToTensor(handle, &slot).
    mlir::LLVM::CallOp::create(builder, loc, unpackFn, {handle, slot});

    // Load the AtenTensorHandle and convert to i64 for the dispatcher.
    mlir::Value atenHandle =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, slot);
    return mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, atenHandle)
        .getResult();
  }
  case c10::TypeKind::BoolType: {
    // Bool: TVMFFIAny payload is already i64 (0 or 1).
    return mlir::LLVM::ExtractValueOp::create(builder, loc, input,
                                              llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType: {
    // Int/SymInt: TVMFFIAny payload is already i64.
    return mlir::LLVM::ExtractValueOp::create(builder, loc, input,
                                              llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType: {
    // Float: TVMFFIAny payload is bitcast f64→i64.
    return mlir::LLVM::ExtractValueOp::create(builder, loc, input,
                                              llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  case c10::TypeKind::NumberType: {
    // Number: inspect the TVMFFIAny type index, then normalize int/float
    // payloads to the dispatcher's double bit-pattern representation.
    mlir::Float64Type f64Ty = mlir::Float64Type::get(ctx);
    mlir::Value typeIndex = mlir::LLVM::ExtractValueOp::create(
        builder, loc, input, llvm::ArrayRef<int64_t>{0});
    mlir::Value payload = mlir::LLVM::ExtractValueOp::create(
        builder, loc, input, llvm::ArrayRef<int64_t>{2});
    mlir::Value intTypeIndex =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, kTVMFFIInt);
    mlir::Value isInt = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, typeIndex, intTypeIndex);
    mlir::Value intAsDouble =
        mlir::LLVM::SIToFPOp::create(builder, loc, f64Ty, payload);
    mlir::Value intPayload =
        mlir::LLVM::BitcastOp::create(builder, loc, i64Ty, intAsDouble);
    return mlir::LLVM::SelectOp::create(builder, loc, isInt, intPayload,
                                        payload)
        .getResult();
  }
  case c10::TypeKind::NoneType: {
    // None: TVMFFIAny payload is 0.
    return mlir::LLVM::ExtractValueOp::create(builder, loc, input,
                                              llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  case c10::TypeKind::OptionalType: {
    // Optional: determine None vs Some from TVMFFIAny TypeIndex (field 0),
    // not from torchType metadata.  Some values are boxed on the heap so
    // that None (0) is always distinguishable from Some(0).
    c10::TypePtr innerC10Type =
        c10Type->cast<c10::OptionalType>()->getElementType();

    // Extract TypeIndex from TVMFFIAny and compare to kTVMFFINone.
    mlir::Value typeIndex = mlir::LLVM::ExtractValueOp::create(
        builder, loc, input, llvm::ArrayRef<int64_t>{0});
    mlir::Value noneIdx =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, kTVMFFINone);
    mlir::Value isNone = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, typeIndex, noneIdx);

    // Split into none / some / continuation blocks.
    // The continuation block takes the result as a block argument (i64).
    mlir::Block *origBlock = builder.getBlock();
    mlir::Block *continuationBlock =
        origBlock->splitBlock(builder.getInsertionPoint());
    continuationBlock->addArgument(i64Ty, loc);

    mlir::Block *noneBlock = builder.createBlock(continuationBlock);
    mlir::Block *someBlock = builder.createBlock(continuationBlock);

    builder.setInsertionPointToEnd(origBlock);
    mlir::LLVM::CondBrOp::create(builder, loc, isNone, noneBlock, someBlock);

    // --- None block: zero (None sentinel) ---
    builder.setInsertionPointToStart(noneBlock);
    mlir::Value zero = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
    mlir::LLVM::BrOp::create(builder, loc, {zero}, continuationBlock);

    // --- Some block: box inner StableIValue on the heap ---
    builder.setInsertionPointToStart(someBlock);
    mlir::Value converted = TRIDENT_UNWRAP_FAILURE(
        buildStableIValue(builder, innerC10Type, input, moduleOp, loc));
    mlir::LLVM::LLVMFuncOp mallocFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateMalloc(moduleOp));
    mlir::Value size = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 8);
    mlir::Value ptr =
        mlir::LLVM::CallOp::create(builder, loc, mallocFn, size).getResult();
    mlir::LLVM::StoreOp::create(builder, loc, converted, ptr);
    mlir::Value boxed =
        mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, ptr);
    mlir::LLVM::BrOp::create(builder, loc, {boxed}, continuationBlock);

    // --- Continuation block (merge point) ---
    builder.setInsertionPointToStart(continuationBlock);
    return continuationBlock->getArgument(0);
  }
  case c10::TypeKind::DeviceObjType: {
    // Device: adapted type is TVMFFIAny with TVM FFI encoding:
    // (device_index << 32) | device_type.
    // The dispatcher expects StableIValue format: (device_type << 32) |
    // device_index.  Convert from TVMFFI to StableIValue encoding.
    mlir::Value combined = mlir::LLVM::ExtractValueOp::create(
        builder, loc, input, llvm::ArrayRef<int64_t>{2});

    // TVM FFI encoding decomposition: low 32 = device_type, high 32 = idx.
    mlir::Value devType =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, combined);
    mlir::Value shift = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 32);
    mlir::Value shifted =
        mlir::LLVM::LShrOp::create(builder, loc, combined, shift);
    mlir::Value devIdx =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, shifted);

    // Encode in StableIValue format: (device_type << 32) | device_index.
    mlir::Value devType64 =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, devType);
    mlir::Value devIdx64 =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, devIdx);
    mlir::Value shiftedType =
        mlir::LLVM::ShlOp::create(builder, loc, devType64, shift);
    return mlir::LLVM::OrOp::create(builder, loc, shiftedType, devIdx64)
        .getResult();
  }
  case c10::TypeKind::ListType: {
    // List: adapted type is TVMFFIAny with payload = TVMFFIObjectHandle
    // (pointer to ffi.Array). Extract pointer, iterate over elements,
    // and recursively convert each element via buildStableIValue.

    c10::TypePtr elemType = c10Type->cast<c10::ListType>()->getElementType();

    mlir::Value inputInt = mlir::LLVM::ExtractValueOp::create(
        builder, loc, input, llvm::ArrayRef<int64_t>{2});
    mlir::LLVM::LLVMStructType anyTy =
        trident::conversion::utils::getTVMFFIAnyType(ctx);

    // --- Step 1: Get list size via ffi.ArraySize ---

    // Build TVMFFIAny(kTVMFFIArray=71, v_obj=input).
    mlir::Value sizeArgSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    mlir::LLVM::StoreOp::create(
        builder, loc, buildFFIAnyValue(builder, loc, kTVMFFIArray, inputInt),
        sizeArgSlot);
    mlir::Value sizeResultSlot = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::callTVMFFIGlobalFunction(
            builder, loc, moduleOp, "ffi.ArraySize", {sizeArgSlot}));
    // Extract v_int64 (field[2]) from result TVMFFIAny.
    mlir::Value listSize = mlir::LLVM::LoadOp::create(
        builder, loc, i64Ty,
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, sizeResultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));

    // --- Step 2: Create StableList with reserved capacity ---

    mlir::Value stableListSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    mlir::LLVM::LLVMFuncOp reserveFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateTorchNewListReserveSize(
            moduleOp));
    mlir::LLVM::CallOp::create(builder, loc, reserveFn,
                               {listSize, stableListSlot});
    mlir::Value listHandle =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, stableListSlot);

    // --- Step 3: Get push_back function handle ---

    mlir::LLVM::LLVMFuncOp pushBackFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateTorchListPushBack(moduleOp));

    // --- Step 4: Loop i = 0 .. listSize-1, ffi.ArrayGetItem + push_back ---

    // Pre-build itemArgs[0] (kTVMFFIArray=71, v_obj=input) — reused across
    // iterations.
    mlir::Value itemArg0 = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    mlir::LLVM::StoreOp::create(
        builder, loc, buildFFIAnyValue(builder, loc, kTVMFFIArray, inputInt),
        itemArg0);

    // Split block — continuationBlock gets the rest of the function.
    mlir::Block *origBlock = builder.getBlock();
    mlir::Block *continuationBlock =
        origBlock->splitBlock(builder.getInsertionPoint());

    // Create loop blocks before continuationBlock.
    // Each takes the loop counter as a block argument (i64).
    mlir::Block *checkBlock = builder.createBlock(continuationBlock);
    checkBlock->addArgument(i64Ty, loc);
    mlir::Block *bodyBlock = builder.createBlock(continuationBlock);
    bodyBlock->addArgument(i64Ty, loc);
    mlir::Block *exitBlock = builder.createBlock(continuationBlock);

    // Branch from origBlock to checkBlock with initial counter = 0.
    builder.setInsertionPointToEnd(origBlock);
    mlir::Value c0 = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
    mlir::LLVM::BrOp::create(builder, loc, {c0}, checkBlock);

    // checkBlock(%iVal : i64)
    builder.setInsertionPointToStart(checkBlock);
    mlir::Value iVal = checkBlock->getArgument(0);
    mlir::Value cond = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::slt, iVal, listSize);
    mlir::LLVM::CondBrOp::create(builder, loc, cond, bodyBlock, {iVal},
                                 exitBlock, {});

    // bodyBlock(%iVal : i64): call ffi.ArrayGetItem, recursively convert
    // element via buildStableIValue, then push_back.
    builder.setInsertionPointToStart(bodyBlock);
    iVal = bodyBlock->getArgument(0);

    // Build itemArgs[1] = TVMFFIAny(kTVMFFIInt=1, v_int64=iVal).
    mlir::Value itemArg1 = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    mlir::LLVM::StoreOp::create(
        builder, loc, buildFFIAnyValue(builder, loc, kTVMFFIInt, iVal),
        itemArg1);

    // Call ffi.ArrayGetItem(array, i).
    mlir::Value itemResult = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::callTVMFFIGlobalFunction(
            builder, loc, moduleOp, "ffi.ArrayGetItem", {itemArg0, itemArg1}));

    // Load the returned TVMFFIAny value (the element).
    mlir::Value elemAny =
        mlir::LLVM::LoadOp::create(builder, loc, anyTy, itemResult);

    // Recursively convert the element using buildStableIValue.
    mlir::Value elemIVal = TRIDENT_UNWRAP_FAILURE(
        buildStableIValue(builder, elemType, elemAny, moduleOp, loc));

    mlir::LLVM::CallOp::create(builder, loc, pushBackFn,
                               {listHandle, elemIVal});

    mlir::Value iPlus1 = mlir::LLVM::AddOp::create(
        builder, loc, iVal,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    mlir::LLVM::BrOp::create(builder, loc, {iPlus1}, checkBlock);

    // exitBlock: br to continuation.
    builder.setInsertionPointToStart(exitBlock);
    mlir::LLVM::BrOp::create(builder, loc, continuationBlock);

    builder.setInsertionPointToStart(continuationBlock);
    return mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, listHandle)
        .getResult();
  }
  default:
    return mlir::failure();
  }
}

/// Converts a type-erased StableIValue (uint64_t) back to the appropriate
/// adapted (LLVM-typed) MLIR Value based on the c10 type from the operator
/// schema.
///
/// \param builder  The MLIR op builder.
/// \param c10Type  The c10 type from the op schema result.
/// \param loaded   The i64 StableIValue loaded from the dispatcher stack.
/// \param moduleOp The parent MLIR module.
/// \param loc      The source location.
/// \return The adapted LLVM-typed MLIR value on success, or failure.
mlir::FailureOr<mlir::Value> resolveStableIValue(mlir::OpBuilder &builder,
                                                 const c10::TypePtr &c10Type,
                                                 mlir::Value loaded,
                                                 mlir::ModuleOp moduleOp,
                                                 mlir::Location loc) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy =
      trident::conversion::utils::getTVMFFIAnyType(ctx);

  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType: {
    // Tensor: i64 → AtenTensorHandle → TVMFFIObjectHandle → TVMFFIAny.
    // The dispatcher returns an AtenTensorHandle in the StableIValue slot;
    // we pack it back into a TVMFFIAny with payload = TVMFFIObjectHandle.

    // Convert i64 back to AtenTensorHandle pointer.
    mlir::Value atenHandle =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, loaded);

    mlir::LLVM::LLVMFuncOp packFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateTensorToTVMFFIObject(moduleOp));

    // Allocate stack slot for TVMFFIObjectHandle output.
    mlir::Value outSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    // Call mTridentTensorToTVMFFIObject(atenHandle, &outSlot).
    mlir::LLVM::CallOp::create(builder, loc, packFn, {atenHandle, outSlot});

    // Load the TVMFFIObjectHandle and build TVMFFIAny {kTVMFFITensor, 0, ptr}.
    mlir::Value handle =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, outSlot);
    mlir::Value ptrI64 =
        mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, handle);
    return buildFFIAnyValue(builder, loc, kTVMFFITensor, ptrI64);
  }
  case c10::TypeKind::BoolType: {
    // Bool: TVMFFIAny {kTVMFFIBool, 0, loaded}.
    return buildFFIAnyValue(builder, loc, kTVMFFIBool, loaded);
  }
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType: {
    // Int/SymInt: TVMFFIAny {kTVMFFIInt, 0, loaded}.
    return buildFFIAnyValue(builder, loc, kTVMFFIInt, loaded);
  }
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType: {
    // Float: TVMFFIAny {kTVMFFIFloat, 0, loaded}.
    // loaded is the i64 bitcast of f64.
    return buildFFIAnyValue(builder, loc, kTVMFFIFloat, loaded);
  }
  case c10::TypeKind::NoneType: {
    // None: TVMFFIAny {kTVMFFINone, 0, 0}.
    mlir::Value zero = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
    return buildFFIAnyValue(builder, loc, kTVMFFINone, zero);
  }
  case c10::TypeKind::OptionalType: {
    // Reverse of buildStableIValue OptionalType encoding:
    // loaded == 0 → None; loaded != 0 → heap pointer to inner StableIValue.
    c10::TypePtr innerC10Type =
        c10Type->cast<c10::OptionalType>()->getElementType();

    // Split blocks: noneBlock (loaded == 0), someBlock (loaded != 0),
    // continuationBlock (merge with TVMFFIAny result).
    mlir::Block *origBlock = builder.getBlock();
    mlir::Block *continuationBlock =
        origBlock->splitBlock(builder.getInsertionPoint());
    continuationBlock->addArgument(anyTy, loc);

    mlir::Block *noneBlock = builder.createBlock(continuationBlock);
    mlir::Block *someBlock = builder.createBlock(continuationBlock);

    mlir::Value zero = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
    mlir::Value isNone = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loaded, zero);

    builder.setInsertionPointToEnd(origBlock);
    mlir::LLVM::CondBrOp::create(builder, loc, isNone, noneBlock, someBlock);

    // --- None block: TVMFFIAny {kTVMFFINone, 0, 0} ---
    builder.setInsertionPointToStart(noneBlock);
    mlir::Value noneResult = buildFFIAnyValue(builder, loc, kTVMFFINone, zero);
    mlir::LLVM::BrOp::create(builder, loc, {noneResult}, continuationBlock);

    // --- Some block: unbox inner StableIValue from heap ---
    builder.setInsertionPointToStart(someBlock);
    mlir::Value heapPtr =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, loaded);
    mlir::Value innerIVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, heapPtr);

    // Free the heap-allocated box.
    mlir::LLVM::LLVMFuncOp freeFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateFree(moduleOp));
    mlir::LLVM::CallOp::create(builder, loc, freeFn, {heapPtr});

    // Recursively resolve the inner StableIValue.
    mlir::Value resolved = TRIDENT_UNWRAP_FAILURE(
        resolveStableIValue(builder, innerC10Type, innerIVal, moduleOp, loc));
    mlir::LLVM::BrOp::create(builder, loc, {resolved}, continuationBlock);

    // --- Continuation block (merge point) ---
    builder.setInsertionPointToStart(continuationBlock);
    return continuationBlock->getArgument(0);
  }
  case c10::TypeKind::DeviceObjType: {
    // Device: dispatcher returns StableIValue format
    // (device_type << 32) | device_index.
    // Convert to TVM FFI encoding: (device_index << 32) | device_type.
    // StableIValue: lower 32 = device_index, upper 32 = device_type.
    mlir::Value devIdx =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, loaded);
    mlir::Value shift = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 32);
    mlir::Value shifted =
        mlir::LLVM::LShrOp::create(builder, loc, loaded, shift);
    mlir::Value devType =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, shifted);

    // Encode in TVM FFI format: (device_index << 32) | device_type.
    mlir::Value devIdx64 =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, devIdx);
    mlir::Value devType64 =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, devType);
    mlir::Value shiftedIdx =
        mlir::LLVM::ShlOp::create(builder, loc, devIdx64, shift);
    mlir::Value combined =
        mlir::LLVM::OrOp::create(builder, loc, shiftedIdx, devType64);
    return buildFFIAnyValue(builder, loc, kTVMFFIDevice, combined);
  }
  case c10::TypeKind::ListType: {
    // Reverse of buildStableIValue ListType encoding:
    // loaded is StableListHandle (i64).  Extract elements via Stable C API
    // and reconstruct a TVM FFI Array.
    c10::TypePtr elemType = c10Type->cast<c10::ListType>()->getElementType();

    // Convert i64 → StableListHandle.
    mlir::Value listHandle =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, loaded);

    // --- Step 1: Get list size via torch_list_size ---
    mlir::Value sizeSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    mlir::LLVM::LLVMFuncOp sizeFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateTorchListSize(moduleOp));
    mlir::LLVM::CallOp::create(builder, loc, sizeFn, {listHandle, sizeSlot});
    mlir::Value listSize =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, sizeSlot);

    // --- Step 2: Allocate ffi.Array args region (runtime size) ---
    mlir::Value ffiArgs =
        mlir::LLVM::AllocaOp::create(builder, loc, ptrTy, anyTy, listSize);

    // --- Step 3: Loop i = 0 .. listSize-1, torch_list_get_item + resolve ---
    mlir::Block *origBlock = builder.getBlock();
    mlir::Block *continuationBlock =
        origBlock->splitBlock(builder.getInsertionPoint());

    mlir::Block *checkBlock = builder.createBlock(continuationBlock);
    checkBlock->addArgument(i64Ty, loc);
    mlir::Block *bodyBlock = builder.createBlock(continuationBlock);
    bodyBlock->addArgument(i64Ty, loc);
    mlir::Block *exitBlock = builder.createBlock(continuationBlock);

    builder.setInsertionPointToEnd(origBlock);
    mlir::Value c0 = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
    mlir::LLVM::BrOp::create(builder, loc, {c0}, checkBlock);

    // checkBlock(%iVal : i64)
    builder.setInsertionPointToStart(checkBlock);
    mlir::Value iVal = checkBlock->getArgument(0);
    mlir::Value cond = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::slt, iVal, listSize);
    mlir::LLVM::CondBrOp::create(builder, loc, cond, bodyBlock, {iVal},
                                 exitBlock, {});

    // bodyBlock(%iVal : i64): extract element, resolve, store to ffiArgs.
    builder.setInsertionPointToStart(bodyBlock);
    iVal = bodyBlock->getArgument(0);

    // Allocate slot for torch_list_get_item output.
    mlir::Value itemSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    mlir::LLVM::LLVMFuncOp getItemFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateTorchListGetItem(moduleOp));
    mlir::LLVM::CallOp::create(builder, loc, getItemFn,
                               {listHandle, iVal, itemSlot});
    mlir::Value itemIVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, itemSlot);

    // Recursively resolve element StableIValue → TVMFFIAny.
    mlir::Value elemAny = TRIDENT_UNWRAP_FAILURE(
        resolveStableIValue(builder, elemType, itemIVal, moduleOp, loc));

    // Store resolved element into ffiArgs[i].
    mlir::Value dstSlot =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, ffiArgs,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{iVal});
    mlir::LLVM::StoreOp::create(builder, loc, elemAny, dstSlot);

    mlir::Value iPlus1 = mlir::LLVM::AddOp::create(
        builder, loc, iVal,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    mlir::LLVM::BrOp::create(builder, loc, {iPlus1}, checkBlock);

    // exitBlock: call ffi.Array with runtime N, then delete StableList.
    builder.setInsertionPointToStart(exitBlock);

    // Truncate listSize to i32 for TVMFFIFunctionCall's numArgs parameter.
    mlir::Value numArgs =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, listSize);
    mlir::Value ffiResult = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::callTVMFFIGlobalFunction(
            builder, loc, moduleOp, "ffi.Array", ffiArgs, numArgs));

    // --- Delete the StableList via torch_delete_list ---
    mlir::LLVM::LLVMFuncOp deleteFn = TRIDENT_UNWRAP_FAILURE(
        trident::conversion::utils::getOrCreateTorchDeleteList(moduleOp));
    mlir::LLVM::CallOp::create(builder, loc, deleteFn, {listHandle});

    // Extract v_obj from result TVMFFIAny and build kTVMFFIArray wrapper.
    mlir::Value vObjGEP =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, ffiResult,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value vObj = mlir::LLVM::LoadOp::create(builder, loc, i64Ty, vObjGEP);

    mlir::LLVM::BrOp::create(builder, loc, continuationBlock);

    // --- Continuation block: emit result TVMFFIAny ---
    builder.setInsertionPointToStart(continuationBlock);
    return buildFFIAnyValue(builder, loc, kTVMFFIArray, vObj);
  }
  default:
    return mlir::failure();
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public C API entry point
//===----------------------------------------------------------------------===//

int TridentSchemaDispatchTorchAtenOp(MlirOperation op, MlirValue *operands,
                                     MlirValue *results,
                                     MlirConversionPatternRewriter rewriter) {
  // Unwrap the C API types.
  mlir::Operation *mlirOp = unwrap(op);
  mlir::ConversionPatternRewriter &mlirRewriter = *unwrap(rewriter);
  mlir::Location loc = mlirOp->getLoc();
  mlir::MLIRContext *ctx = mlirOp->getContext();

  // Extract the op name.
  llvm::StringRef opName = mlirOp->getName().getStringRef();

  // Match "torch.aten.<OpName>[.<Overload>]" using std::regex.
  static const std::regex re(
      R"(^torch\.aten\.([_a-zA-Z]+)(?:\.([_a-zA-Z]+))?$)");
  std::smatch m;
  std::string opNameCopy = opName.str();
  if (!std::regex_match(opNameCopy, m, re)) {
    return 1;
  }

  const std::string atenOpName = m[1].str();
  const std::string dispatcherOpName = "aten::" + atenOpName;
  const std::string dispatcherOverloadName = m[2].str();

  // Look up the operator schema via c10::Dispatcher.
  c10::OperatorName c10OpName(dispatcherOpName, dispatcherOverloadName);
  std::optional<at::OperatorHandle> opHandle =
      c10::Dispatcher::singleton().findSchema(c10OpName);
  if (!opHandle.has_value()) {
    mlirOp->emitError() << "operator " << dispatcherOpName << "."
                        << dispatcherOverloadName
                        << " which doesn't have a schema registered yet";
    return 1;
  }

  const at::FunctionSchema &schema = opHandle->schema();
  const std::vector<at::Argument> &schemaArgs = schema.arguments();
  const std::vector<at::Argument> &schemaReturns = schema.returns();

  // Find the parent module.
  mlir::ModuleOp moduleOp = mlirOp->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp) {
    mlirOp->emitError("op is not inside a module");
    return 1;
  }

  const size_t numInputs = mlirOp->getNumOperands();
  const size_t numResults = mlirOp->getNumResults();

  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::Float64Type f64Ty = mlir::Float64Type::get(ctx);

  if (mlir::isa<mlir::torch::Torch::AtenSubScalarOp>(mlirOp)) {
    if (numInputs != 3 || numResults != 1) {
      mlirOp->emitError(
          "AtenSubScalarOp specialization expects 3 inputs and 1 result");
      return 1;
    }
    mlir::Value selfDecoded = TRIDENT_UNWRAP(
        buildShimTensorValue(mlirRewriter, loc, unwrap(operands[0]), moduleOp),
        {
          mlirOp->emitError("failed to decode tensor argument");
          return 1;
        });
    mlir::Value otherDecoded =
        buildShimNumberValue(mlirRewriter, loc, unwrap(operands[1]));
    mlir::Value alphaDecoded =
        buildShimNumberValue(mlirRewriter, loc, unwrap(operands[2]));
    // `aten::sub.Scalar(self, other, alpha)` is lowered via
    // `aoti_torch_cuda_add_Scalar(self, other, alpha, out)`, so `other`
    // must be negated first to preserve subtraction semantics.
    mlir::Value negatedOther =
        mlir::LLVM::FNegOp::create(mlirRewriter, loc, otherDecoded).getResult();
    mlir::Value retSlot = mlir::LLVM::AllocaOp::create(
        mlirRewriter, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(mlirRewriter, loc, i64Ty, 1));

    mlir::LLVM::LLVMFuncOp shimFn = TRIDENT_UNWRAP(
        trident::conversion::utils::getOrCreateAOTITorchCudaAddScalar(moduleOp),
        {
          mlirOp->emitError("failed to declare shim function "
                            "aoti_torch_cuda_add_Scalar");
          return 1;
        });
    mlir::LLVM::CallOp::create(
        mlirRewriter, loc, shimFn,
        {selfDecoded, negatedOther, alphaDecoded, retSlot});
    mlir::Value rawRet =
        mlir::LLVM::LoadOp::create(mlirRewriter, loc, ptrTy, retSlot);
    mlir::Value resolved = TRIDENT_UNWRAP(
        resolveShimTensorValue(mlirRewriter, loc, rawRet, moduleOp), {
          mlirOp->emitError("failed to encode specialization return value");
          return 1;
        });
    results[0] = wrap(resolved);
  } else if (mlir::isa<mlir::torch::Torch::AtenSubTensorOp>(mlirOp)) {
    if (numInputs != 3 || numResults != 1) {
      mlirOp->emitError(
          "AtenSubTensorOp specialization expects 3 inputs and 1 result");
      return 1;
    }
    mlir::Value selfDecoded = TRIDENT_UNWRAP(
        buildShimTensorValue(mlirRewriter, loc, unwrap(operands[0]), moduleOp),
        {
          mlirOp->emitError("failed to decode lhs tensor argument");
          return 1;
        });
    mlir::Value otherDecoded = TRIDENT_UNWRAP(
        buildShimTensorValue(mlirRewriter, loc, unwrap(operands[1]), moduleOp),
        {
          mlirOp->emitError("failed to decode rhs tensor argument");
          return 1;
        });
    mlir::Value alphaDecoded =
        buildShimNumberValue(mlirRewriter, loc, unwrap(operands[2]));
    mlir::Value retSlot = mlir::LLVM::AllocaOp::create(
        mlirRewriter, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(mlirRewriter, loc, i64Ty, 1));
    mlir::LLVM::LLVMFuncOp shimFn = TRIDENT_UNWRAP(
        trident::conversion::utils::getOrCreateAOTITorchAtenSubtractTensor(
            moduleOp),
        {
          mlirOp->emitError("failed to declare shim function "
                            "aoti_torch_aten_subtract_Tensor");
          return 1;
        });
    mlir::LLVM::CallOp::create(
        mlirRewriter, loc, shimFn,
        {selfDecoded, otherDecoded, alphaDecoded, retSlot});
    mlir::Value rawRet =
        mlir::LLVM::LoadOp::create(mlirRewriter, loc, ptrTy, retSlot);

    mlir::Value resolved = TRIDENT_UNWRAP(
        resolveShimTensorValue(mlirRewriter, loc, rawRet, moduleOp), {
          mlirOp->emitError("failed to encode specialization return value");
          return 1;
        });
    results[0] = wrap(resolved);
  } else if (llvm::any_of(schemaArgs, [](const at::Argument &arg) {
               return arg.type()->kind() == c10::TypeKind::NumberType;
             })) {
    if (numInputs != schemaArgs.size() || numResults != schemaReturns.size()) {
      mlirOp->emitError("operand or result count does not match schema arity");
      return 1;
    }
    // NumberType/Scalar path: Stable C API dispatcher does not support Scalar
    // arguments. Route to torchgen-generated CUDA shim symbols instead.
    // TODO: In future, dynamically obtain the device prefix at runtime
    // instead of hardcoding "_cuda_", while assume the current device is
    // definitely CUDA currently
    llvm::SmallVector<mlir::Type> shimArgTypes;
    llvm::SmallVector<mlir::Value> shimArgs;

    for (auto [idx, schemaArg] : llvm::enumerate(schemaArgs)) {
      c10::TypePtr argType = schemaArg.type();
      mlir::Value operand = unwrap(operands[idx]);
      mlir::Value value = TRIDENT_UNWRAP(
          buildShimValue(mlirRewriter, loc, argType, operand, moduleOp), {
            mlirOp->emitError("failed to decode shim argument for type: ")
                << c10::typeKindToString(argType->kind());
            return 1;
          });

      shimArgTypes.push_back(value.getType());
      shimArgs.push_back(value);
    }

    llvm::SmallVector<mlir::Type> retElemTypes;
    llvm::SmallVector<mlir::Value> retSlots;
    for (const at::Argument &schemaReturn : schemaReturns) {
      c10::TypePtr retType = schemaReturn.type();
      mlir::Type retElemTy = getShimElemType(ctx, retType);
      if (!retElemTy) {
        mlirOp->emitError("unsupported NumberType shim result type: ")
            << c10::typeKindToString(retType->kind());
        return 1;
      }

      mlir::Value retSlot = mlir::LLVM::AllocaOp::create(
          mlirRewriter, loc, ptrTy, retElemTy,
          mlir::LLVM::ConstantOp::create(mlirRewriter, loc, i64Ty, 1));
      shimArgTypes.push_back(ptrTy);
      shimArgs.push_back(retSlot);
      retElemTypes.push_back(retElemTy);
      retSlots.push_back(retSlot);
    }

    std::string shimSymbol = "aoti_torch_cuda_" + atenOpName;
    if (!dispatcherOverloadName.empty()) {
      shimSymbol += "_" + dispatcherOverloadName;
    }

    mlir::Type errTy = mlir::IntegerType::get(ctx, 32);
    mlir::LLVM::LLVMFunctionType shimFnTy =
        mlir::LLVM::LLVMFunctionType::get(errTy, shimArgTypes);
    mlir::LLVM::LLVMFuncOp shimFn = TRIDENT_UNWRAP(
        trident::conversion::utils::getOrCreateCAPI(moduleOp, shimSymbol,
                                                    shimFnTy),
        {
          mlirOp->emitError()
              << "failed to declare shim function " << shimSymbol;
          return 1;
        });

    mlir::LLVM::CallOp::create(mlirRewriter, loc, shimFn, shimArgs);

    for (auto [idx, tuple] :
         llvm::enumerate(llvm::zip(schemaReturns, retElemTypes, retSlots))) {
      auto [schemaReturn, retElemTy, retSlot] = tuple;
      mlir::Value rawRet =
          mlir::LLVM::LoadOp::create(mlirRewriter, loc, retElemTy, retSlot);
      mlir::Value resolved = TRIDENT_UNWRAP(
          resolveShimValue(mlirRewriter, loc, schemaReturn.type(), rawRet,
                           moduleOp),
          {
            mlirOp->emitError("failed to encode shim return value");
            return 1;
          });
      results[idx] = wrap(resolved);
    }
  } else {
    if (numInputs != schemaArgs.size() || numResults != schemaReturns.size()) {
      mlirOp->emitError("operand or result count does not match schema arity");
      return 1;
    }

    const size_t maxCount = std::max(numInputs, numResults);

    // Allocate an i64 slot array with maxCount elements on the stack.
    mlir::Value array = mlir::LLVM::AllocaOp::create(
        mlirRewriter, loc, ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(mlirRewriter, loc, i64Ty, maxCount)
            .getResult());

    // Convert and store each operand into the slot array.
    for (auto [idx, schemaArg] : llvm::enumerate(schemaArgs)) {
      mlir::Value operand = unwrap(operands[idx]);

      c10::TypePtr argType = schemaArg.type();
      mlir::Value built = TRIDENT_UNWRAP(
          buildStableIValue(mlirRewriter, argType, operand, moduleOp, loc), {
            mlirOp->emitError("unsupported input type: ")
                << c10::typeKindToString(argType->kind());
            return 1;
          });
      mlir::Value ival = built;

      mlir::Value ptr = mlir::LLVM::GEPOp::create(mlirRewriter, loc, ptrTy,
                                                  i64Ty, array, {idx});
      mlir::LLVM::StoreOp::create(mlirRewriter, loc, ival, ptr);
    }

    // Get or create the aoti_torch_call_dispatcher function declaration.
    mlir::LLVM::LLVMFuncOp calleeOrErr = TRIDENT_UNWRAP(
        trident::conversion::utils::getOrCreateAOTITorchCallDispatcher(
            moduleOp),
        return 1);

    // Create global string constants for op_name and overload_name.
    mlir::Value opNamePtr = trident::conversion::utils::getOrCreateGlobalString(
        mlirRewriter, loc, moduleOp, "op", dispatcherOpName);
    mlir::Value overloadNamePtr =
        trident::conversion::utils::getOrCreateGlobalString(
            mlirRewriter, loc, moduleOp, "overload", dispatcherOverloadName);

    // Call aoti_torch_call_dispatcher(opName, overloadName, slotArray).
    mlir::LLVM::CallOp::create(mlirRewriter, loc, calleeOrErr,
                               {opNamePtr, overloadNamePtr, array});

    // Load and convert each result from the slot array.
    for (size_t i = 0; i < numResults; ++i) {
      mlir::Value ptr = mlir::LLVM::GEPOp::create(mlirRewriter, loc, ptrTy,
                                                  i64Ty, array, {i});
      mlir::Value loaded =
          mlir::LLVM::LoadOp::create(mlirRewriter, loc, i64Ty, ptr);

      c10::TypePtr retType = schemaReturns[i].type();
      mlir::Value resolved = TRIDENT_UNWRAP(
          resolveStableIValue(mlirRewriter, retType, loaded, moduleOp, loc), {
            mlirOp->emitError("unsupported result type: ")
                << c10::typeKindToString(retType->kind());
            return 1;
          });
      results[i] = wrap(resolved);
    }
  }

  return 0;
}
