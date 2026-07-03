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
#include "trident-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident-core/Conversion/Utils/GlobalString.h"
#include "trident-core/Conversion/Utils/StableCAPIDescriptors.h"
#include "trident-core/Conversion/Utils/StdLibCAPIDescriptors.h"
#include "trident-core/Conversion/Utils/TVMFFIUtils.h"
#include "trident-core/Conversion/Utils/TridentCAPIDescriptors.h"
#include "trident-core/Conversion/Utils/Type.h"
#include "tvm/ffi/c_api.h"

#include <regex>

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

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

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> unpackFn =
        trident::conversion::utils::getOrCreateTVMFFIObjectToTensor(moduleOp);
    if (mlir::failed(unpackFn)) {
      return mlir::failure();
    }

    // Allocate stack slot for AtenTensorHandle output.
    mlir::Value slot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    // Call mTridentTVMFFIObjectToTensor(handle, &slot).
    mlir::LLVM::CallOp::create(builder, loc, *unpackFn,
                               mlir::ValueRange{handle, slot});

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
    mlir::LLVM::BrOp::create(builder, loc, mlir::ValueRange{zero},
                             continuationBlock);

    // --- Some block: box inner StableIValue on the heap ---
    builder.setInsertionPointToStart(someBlock);
    mlir::FailureOr<mlir::Value> converted =
        buildStableIValue(builder, innerC10Type, input, moduleOp, loc);
    if (mlir::failed(converted)) {
      return mlir::failure();
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> mallocFn =
        trident::conversion::utils::getOrCreateMalloc(moduleOp);
    if (mlir::failed(mallocFn)) {
      return mlir::failure();
    }
    mlir::Value size = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 8);
    mlir::Value ptr =
        mlir::LLVM::CallOp::create(builder, loc, *mallocFn, size).getResult();
    mlir::LLVM::StoreOp::create(builder, loc, *converted, ptr);
    mlir::Value boxed =
        mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, ptr);
    mlir::LLVM::BrOp::create(builder, loc, mlir::ValueRange{boxed},
                             continuationBlock);

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
    mlir::FailureOr<mlir::Value> sizeResultSlot =
        trident::conversion::utils::callTVMFFIGlobalFunction(
            builder, loc, moduleOp, "ffi.ArraySize", {sizeArgSlot});
    if (mlir::failed(sizeResultSlot)) {
      return mlir::failure();
    }
    // Extract v_int64 (field[2]) from result TVMFFIAny.
    mlir::Value listSize = mlir::LLVM::LoadOp::create(
        builder, loc, i64Ty,
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, *sizeResultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));

    // --- Step 2: Create StableList with reserved capacity ---

    mlir::Value stableListSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> reserveFn =
        trident::conversion::utils::getOrCreateTorchNewListReserveSize(
            moduleOp);
    if (mlir::failed(reserveFn)) {
      return mlir::failure();
    }
    mlir::LLVM::CallOp::create(builder, loc, *reserveFn,
                               {listSize, stableListSlot});
    mlir::Value listHandle =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, stableListSlot);

    // --- Step 3: Get push_back function handle ---

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> pushBackFn =
        trident::conversion::utils::getOrCreateTorchListPushBack(moduleOp);
    if (mlir::failed(pushBackFn)) {
      return mlir::failure();
    }

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
    mlir::FailureOr<mlir::Value> itemResult =
        trident::conversion::utils::callTVMFFIGlobalFunction(
            builder, loc, moduleOp, "ffi.ArrayGetItem", {itemArg0, itemArg1});
    if (mlir::failed(itemResult)) {
      return mlir::failure();
    }

    // Load the returned TVMFFIAny value (the element).
    mlir::Value elemAny =
        mlir::LLVM::LoadOp::create(builder, loc, anyTy, *itemResult);

    // Recursively convert the element using buildStableIValue.
    mlir::FailureOr<mlir::Value> elemIVal =
        buildStableIValue(builder, elemType, elemAny, moduleOp, loc);
    if (mlir::failed(elemIVal)) {
      return mlir::failure();
    }

    mlir::LLVM::CallOp::create(builder, loc, *pushBackFn,
                               {listHandle, *elemIVal});

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

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> packFn =
        trident::conversion::utils::getOrCreateTensorToTVMFFIObject(moduleOp);
    if (mlir::failed(packFn)) {
      return mlir::failure();
    }

    // Allocate stack slot for TVMFFIObjectHandle output.
    mlir::Value outSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    // Call mTridentTensorToTVMFFIObject(atenHandle, &outSlot).
    mlir::LLVM::CallOp::create(builder, loc, *packFn,
                               mlir::ValueRange{atenHandle, outSlot});

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
    mlir::LLVM::BrOp::create(builder, loc, mlir::ValueRange{noneResult},
                             continuationBlock);

    // --- Some block: unbox inner StableIValue from heap ---
    builder.setInsertionPointToStart(someBlock);
    mlir::Value heapPtr =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, loaded);
    mlir::Value innerIVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, heapPtr);

    // Free the heap-allocated box.
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> freeFn =
        trident::conversion::utils::getOrCreateFree(moduleOp);
    if (mlir::failed(freeFn)) {
      return mlir::failure();
    }
    mlir::LLVM::CallOp::create(builder, loc, *freeFn, {heapPtr});

    // Recursively resolve the inner StableIValue.
    mlir::FailureOr<mlir::Value> resolved =
        resolveStableIValue(builder, innerC10Type, innerIVal, moduleOp, loc);
    if (mlir::failed(resolved)) {
      return mlir::failure();
    }
    mlir::LLVM::BrOp::create(builder, loc, mlir::ValueRange{*resolved},
                             continuationBlock);

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
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> sizeFn =
        trident::conversion::utils::getOrCreateTorchListSize(moduleOp);
    if (mlir::failed(sizeFn)) {
      return mlir::failure();
    }
    mlir::LLVM::CallOp::create(builder, loc, *sizeFn, {listHandle, sizeSlot});
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

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> getItemFn =
        trident::conversion::utils::getOrCreateTorchListGetItem(moduleOp);
    if (mlir::failed(getItemFn)) {
      return mlir::failure();
    }
    mlir::LLVM::CallOp::create(builder, loc, *getItemFn,
                               {listHandle, iVal, itemSlot});
    mlir::Value itemIVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, itemSlot);

    // Recursively resolve element StableIValue → TVMFFIAny.
    mlir::FailureOr<mlir::Value> elemAny =
        resolveStableIValue(builder, elemType, itemIVal, moduleOp, loc);
    if (mlir::failed(elemAny)) {
      return mlir::failure();
    }

    // Store resolved element into ffiArgs[i].
    mlir::Value dstSlot =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, ffiArgs,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{iVal});
    mlir::LLVM::StoreOp::create(builder, loc, *elemAny, dstSlot);

    mlir::Value iPlus1 = mlir::LLVM::AddOp::create(
        builder, loc, iVal,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    mlir::LLVM::BrOp::create(builder, loc, {iPlus1}, checkBlock);

    // exitBlock: call ffi.Array with runtime N, then delete StableList.
    builder.setInsertionPointToStart(exitBlock);

    // Truncate listSize to i32 for TVMFFIFunctionCall's numArgs parameter.
    mlir::Value numArgs =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, listSize);
    mlir::FailureOr<mlir::Value> ffiResult =
        trident::conversion::utils::callTVMFFIGlobalFunction(
            builder, loc, moduleOp, "ffi.Array", ffiArgs, numArgs);
    if (mlir::failed(ffiResult)) {
      return mlir::failure();
    }

    // --- Delete the StableList via torch_delete_list ---
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> deleteFn =
        trident::conversion::utils::getOrCreateTorchDeleteList(moduleOp);
    if (mlir::failed(deleteFn)) {
      return mlir::failure();
    }
    mlir::LLVM::CallOp::create(builder, loc, *deleteFn, {listHandle});

    // Extract v_obj from result TVMFFIAny and build kTVMFFIArray wrapper.
    mlir::Value vObjGEP =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, *ffiResult,
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

  const std::string dispatcherOpName = "aten::" + m[1].str();
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
  const size_t maxCount = std::max(numInputs, numResults);

  // Allocate an i64 slot array with maxCount elements on the stack.
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::Value array = mlir::LLVM::AllocaOp::create(
      mlirRewriter, loc, ptrTy, i64Ty,
      mlir::LLVM::ConstantOp::create(mlirRewriter, loc, i64Ty, maxCount)
          .getResult());

  // Convert and store each operand into the slot array.
  for (size_t i = 0; i < numInputs; ++i) {
    mlir::Value adaptedVal = unwrap(operands[i]);

    mlir::Value ival;
    if (i < schemaArgs.size()) {
      c10::TypePtr argType = schemaArgs[i].type();
      mlir::FailureOr<mlir::Value> built =
          buildStableIValue(mlirRewriter, argType, adaptedVal, moduleOp, loc);
      if (mlir::failed(built)) {
        mlirOp->emitError("unsupported input type: ")
            << c10::typeKindToString(argType->kind());
        return 1;
      }
      ival = *built;
    } else {
      // No schema type info: pass the adapted value as-is (assume i64).
      ival = adaptedVal;
    }

    mlir::Value ptr =
        mlir::LLVM::GEPOp::create(mlirRewriter, loc, ptrTy, i64Ty, array, {i});
    mlir::LLVM::StoreOp::create(mlirRewriter, loc, ival, ptr);
  }

  // Get or create the aoti_torch_call_dispatcher function declaration.
  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> calleeOrErr =
      trident::conversion::utils::getOrCreateAOTITorchCallDispatcher(moduleOp);
  if (mlir::failed(calleeOrErr)) {
    return 1;
  }

  // Create global string constants for op_name and overload_name.
  mlir::Value opNamePtr = trident::conversion::utils::getOrCreateGlobalString(
      mlirRewriter, loc, moduleOp, "op", dispatcherOpName);
  mlir::Value overloadNamePtr =
      trident::conversion::utils::getOrCreateGlobalString(
          mlirRewriter, loc, moduleOp, "overload", dispatcherOverloadName);

  // Call aoti_torch_call_dispatcher(opName, overloadName, slotArray).
  mlir::LLVM::CallOp::create(
      mlirRewriter, loc, *calleeOrErr,
      mlir::ValueRange{opNamePtr, overloadNamePtr, array});

  // Load and convert each result from the slot array.
  for (size_t i = 0; i < numResults; ++i) {
    mlir::Value ptr =
        mlir::LLVM::GEPOp::create(mlirRewriter, loc, ptrTy, i64Ty, array, {i});
    mlir::Value loaded =
        mlir::LLVM::LoadOp::create(mlirRewriter, loc, i64Ty, ptr);

    mlir::Value result;
    if (i < schemaReturns.size()) {
      c10::TypePtr retType = schemaReturns[i].type();
      mlir::FailureOr<mlir::Value> resolved =
          resolveStableIValue(mlirRewriter, retType, loaded, moduleOp, loc);
      if (mlir::failed(resolved)) {
        mlirOp->emitError("unsupported result type: ")
            << c10::typeKindToString(retType->kind());
        return 1;
      }
      result = *resolved;
    } else {
      result = loaded;
    }

    results[i] = wrap(result);
  }

  return 0;
}
