//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchemaLookup.h"
#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/function_schema.h"
#include "ATen/core/jit_type_base.h"
#include "c10/core/ScalarType.h"
#include "libtriton-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/GlobalString.h"
#include "libtriton-core/Conversion/Utils/LibTritonCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/StableCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/StdLibCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/TVMFFIUtils.h"
#include "libtriton-core/Conversion/Utils/Type.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#include <regex>

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Fill a pre-allocated TVMFFIAny slot with typeIndex and i64 payload.
static void fillFFIAny(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value elemPtr, int32_t typeIndex,
                       mlir::Value valI64) {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto anyTy =
      libtriton::conversion::utils::getTVMFFIAnyType(builder.getContext());
  mlir::LLVM::StoreOp::create(
      builder, loc,
      mlir::LLVM::ConstantOp::create(
          builder, loc, mlir::IntegerType::get(builder.getContext(), 32),
          typeIndex),
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, elemPtr,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
  mlir::LLVM::StoreOp::create(
      builder, loc,
      mlir::LLVM::ConstantOp::create(
          builder, loc, mlir::IntegerType::get(builder.getContext(), 32), 0),
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, elemPtr,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));
  mlir::LLVM::StoreOp::create(
      builder, loc, valI64,
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, elemPtr,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
}

//===----------------------------------------------------------------------===//
// StableIValue building from c10 type info
//===----------------------------------------------------------------------===//

/// Converts an adapted (LLVM-typed) MLIR Value into a type-erased StableIValue
/// (uint64_t) based on the c10 type from the operator schema.
///
/// \param builder  The MLIR op builder.
/// \param c10Type  The c10 type from the op schema argument.
/// \param torchType The original torch MLIR type of the operand, before LLVM
///                  lowering (e.g. !torch.none, !torch.int).
/// \param input    The adapted LLVM-typed MLIR value.
/// \param moduleOp The parent MLIR module.
/// \param loc      The source location.
/// \return The i64 StableIValue on success, or failure.
mlir::FailureOr<mlir::Value>
buildStableIValue(mlir::OpBuilder &builder, const c10::TypePtr &c10Type,
                  mlir::Type torchType, mlir::Value input,
                  mlir::ModuleOp moduleOp, mlir::Location loc) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType: {
    // Tensor: adapted value is a TVMFFIObjectHandle (!llvm.ptr).
    // aoti_torch_call_dispatcher expects an AtenTensorHandle in the
    // StableIValue slot, so we must unpack the AtenTensorHandle from
    // the TVMFFIObjectHandle first.

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> unpackFn =
        libtriton::conversion::utils::getOrCreateTVMFFIObjectToTensor(moduleOp);
    if (mlir::failed(unpackFn)) {
      return mlir::failure();
    }

    // Allocate stack slot for AtenTensorHandle output.
    mlir::Value slot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    // Call mLibTritonTVMFFIObjectToTensor(handle, &slot).
    mlir::LLVM::CallOp::create(builder, loc, *unpackFn,
                               mlir::ValueRange{input, slot});

    // Load the AtenTensorHandle and convert to i64 for the dispatcher.
    mlir::Value atenHandle =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, slot);
    return mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, atenHandle)
        .getResult();
  }
  case c10::TypeKind::BoolType: {
    // Bool: zero-extend i1 to i64.
    return mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, input).getResult();
  }
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType: {
    // Int/SymInt: already i64, pass through.
    return input;
  }
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType: {
    // Float: bitcast f64 to i64.
    return mlir::LLVM::BitcastOp::create(builder, loc, i64Ty, input)
        .getResult();
  }
  case c10::TypeKind::NoneType: {
    // None: constant 0.
    return mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0).getResult();
  }
  case c10::TypeKind::OptionalType: {
    // Optional: unified CondBr + blocks lowering for all three sub-cases.
    // - torch.none               → flag = constant 0 (always zero branch)
    // - torch.optional<T>        → flag = extractvalue struct[0]
    // - direct type (e.g. int)   → flag = constant 1 (always malloc branch)
    // Dead branches contain dummy operations; MLIR canonicalization removes
    // them later.

    c10::TypePtr innerC10Type =
        c10Type->cast<c10::OptionalType>()->getElementType();

    // Determine the branch condition and inner value metadata.
    mlir::IntegerType i1Ty = mlir::IntegerType::get(ctx, 1);
    mlir::Value flag;
    mlir::Value innerAdapted;
    mlir::Type innerOrigType;
    const bool isNoneType = llvm::isa<mlir::torch::Torch::NoneType>(torchType);

    if (isNoneType) {
      // Always None: flag = 0 (then-block is dead).
      flag = mlir::LLVM::ConstantOp::create(builder, loc, i1Ty, 0);
    } else if (mlir::torch::Torch::OptionalType optTy =
                   llvm::dyn_cast<mlir::torch::Torch::OptionalType>(
                       torchType)) {
      // Optional wrapper: flag is field 0 of llvm.struct<(i1, T)>.
      mlir::LLVM::LLVMStructType structTy =
          llvm::cast<mlir::LLVM::LLVMStructType>(input.getType());
      llvm::ArrayRef<mlir::Type> structBody = structTy.getBody();
      flag = mlir::LLVM::ExtractValueOp::create(
                 builder, loc, structBody[0], input, llvm::ArrayRef<int64_t>{0})
                 .getResult();
      innerAdapted =
          mlir::LLVM::ExtractValueOp::create(builder, loc, structBody[1], input,
                                             llvm::ArrayRef<int64_t>{1})
              .getResult();
      innerOrigType = optTy.getContainedType();
    } else {
      // Direct type: flag = 1 (else-block is dead).
      flag = mlir::LLVM::ConstantOp::create(builder, loc, i1Ty, 1);
      innerAdapted = input;
      innerOrigType = torchType;
    }

    // Allocate a stack slot for the result (must dominate both branches).
    mlir::Value resultSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    // Split the current block; the continuation serves as the merge target.
    mlir::Block *origBlock = builder.getBlock();
    mlir::Block *continuationBlock =
        origBlock->splitBlock(builder.getInsertionPoint());

    mlir::Block *thenBlock = builder.createBlock(continuationBlock);
    mlir::Block *elseBlock = builder.createBlock(continuationBlock);

    // Terminate origBlock with cond_br.
    builder.setInsertionPointToEnd(origBlock);
    mlir::LLVM::CondBrOp::create(builder, loc, flag, thenBlock, elseBlock);

    // --- Then block: box inner StableIValue on the heap ---
    builder.setInsertionPointToStart(thenBlock);
    mlir::Value innerIVal;
    if (isNoneType) {
      // Dead branch: emit a dummy constant (simplified away later).
      innerIVal = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
    } else {
      mlir::FailureOr<mlir::Value> converted = buildStableIValue(
          builder, innerC10Type, innerOrigType, innerAdapted, moduleOp, loc);
      if (mlir::failed(converted)) {
        return mlir::failure();
      }
      innerIVal = *converted;
    }

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> mallocFn =
        libtriton::conversion::utils::getOrCreateMalloc(moduleOp);
    if (mlir::failed(mallocFn)) {
      return mlir::failure();
    }
    mlir::Value size = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 8);
    mlir::Value ptr =
        mlir::LLVM::CallOp::create(builder, loc, *mallocFn, size).getResult();
    mlir::LLVM::StoreOp::create(builder, loc, innerIVal, ptr);
    mlir::Value thenVal =
        mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, ptr);
    mlir::LLVM::StoreOp::create(builder, loc, thenVal, resultSlot);
    mlir::LLVM::BrOp::create(builder, loc, continuationBlock);

    // --- Else block: zero (None sentinel) ---
    builder.setInsertionPointToStart(elseBlock);
    mlir::Value zero = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
    mlir::LLVM::StoreOp::create(builder, loc, zero, resultSlot);
    mlir::LLVM::BrOp::create(builder, loc, continuationBlock);

    // --- Continuation block (merge point) ---
    builder.setInsertionPointToStart(continuationBlock);
    return mlir::LLVM::LoadOp::create(builder, loc, i64Ty, resultSlot)
        .getResult();
  }
  case c10::TypeKind::DeviceObjType: {
    // Device: adapted type is llvm.struct<(i32, i32)>.
    // Pack device_type and device_index into a single i64.
    // PyTorch StableIValue format: (device_type << 32) | device_index
    // Upper 32 bits = device_type, lower 32 bits = device_index.
    mlir::Value devType = mlir::LLVM::ExtractValueOp::create(
        builder, loc, i32Ty, input, llvm::ArrayRef<int64_t>{0});
    mlir::Value devIdx = mlir::LLVM::ExtractValueOp::create(
        builder, loc, i32Ty, input, llvm::ArrayRef<int64_t>{1});
    mlir::Value devType64 =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, devType);
    mlir::Value devIdx64 =
        mlir::LLVM::ZExtOp::create(builder, loc, i64Ty, devIdx);
    mlir::Value shift = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 32);
    mlir::Value shifted =
        mlir::LLVM::ShlOp::create(builder, loc, devType64, shift);
    return mlir::LLVM::OrOp::create(builder, loc, shifted, devIdx64)
        .getResult();
  }
  case c10::TypeKind::ListType: {
    // List: adapted type is !llvm.ptr (TVMFFIObjectHandle pointing to an
    // ffi.Array).  Convert to StableListHandle via TVM FFI C API calls
    // (ffi.ArraySize / ffi.ArrayGetItem) + Stable C API
    // (torch_new_list_reserve_size / torch_list_push_back).

    mlir::LLVM::LLVMStructType anyTy =
        libtriton::conversion::utils::getTVMFFIAnyType(ctx);

    // --- Step 1: Get list size via ffi.ArraySize ---

    mlir::Value inputInt =
        mlir::LLVM::PtrToIntOp::create(builder, loc, i64Ty, input);
    // Build TVMFFIAny(kTVMFFIArray=71, v_obj=input).
    mlir::Value sizeArgSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    fillFFIAny(builder, loc, sizeArgSlot, kTVMFFIArray, inputInt);
    mlir::FailureOr<mlir::Value> sizeResultSlot =
        libtriton::conversion::utils::callTVMFFIGlobalFunction(
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
        libtriton::conversion::utils::getOrCreateTorchNewListReserveSize(
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
        libtriton::conversion::utils::getOrCreateTorchListPushBack(moduleOp);
    if (mlir::failed(pushBackFn)) {
      return mlir::failure();
    }

    // --- Step 4: Loop i = 0 .. listSize-1, ffi.ArrayGetItem + push_back ---

    // Pre-build itemArgs[0] (kTVMFFIArray=71, v_obj=input) — reused across
    // iterations.
    mlir::Value itemArg0 = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    fillFFIAny(builder, loc, itemArg0, kTVMFFIArray, inputInt);

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
    mlir::LLVM::BrOp::create(builder, loc, mlir::ValueRange{c0}, checkBlock);

    // checkBlock(%iVal : i64)
    builder.setInsertionPointToStart(checkBlock);
    mlir::Value iVal = checkBlock->getArgument(0);
    mlir::Value cond = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::slt, iVal, listSize);
    mlir::LLVM::CondBrOp::create(builder, loc, cond, bodyBlock,
                                 mlir::ValueRange{iVal}, exitBlock,
                                 mlir::ValueRange{});

    // bodyBlock(%iVal : i64): call ffi.ArrayGetItem via
    // callTVMFFIGlobalFunction.
    builder.setInsertionPointToStart(bodyBlock);
    iVal = bodyBlock->getArgument(0);

    // Build itemArgs[1] = TVMFFIAny(kTVMFFIInt=1, v_int64=iVal).
    mlir::Value itemArg1 = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    fillFFIAny(builder, loc, itemArg1, kTVMFFIInt, iVal);

    // Call ffi.ArrayGetItem(array, i).
    mlir::FailureOr<mlir::Value> itemResult =
        libtriton::conversion::utils::callTVMFFIGlobalFunction(
            builder, loc, moduleOp, "ffi.ArrayGetItem", {itemArg0, itemArg1});
    if (mlir::failed(itemResult))
      return mlir::failure();
    mlir::Value elemVal = mlir::LLVM::LoadOp::create(
        builder, loc, i64Ty,
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, *itemResult,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));

    mlir::LLVM::CallOp::create(builder, loc, *pushBackFn,
                               {listHandle, elemVal});

    mlir::Value iPlus1 = mlir::LLVM::AddOp::create(
        builder, loc, iVal,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));
    mlir::LLVM::BrOp::create(builder, loc, mlir::ValueRange{iPlus1},
                             checkBlock);

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
/// \param torchType The original torch MLIR type of the result (unused for
///                  non-optional types; passed for symmetry).
/// \param loaded   The i64 StableIValue loaded from the dispatcher stack.
/// \param moduleOp The parent MLIR module.
/// \param loc      The source location.
/// \return The adapted LLVM-typed MLIR value on success, or failure.
mlir::FailureOr<mlir::Value>
resolveStableIValue(mlir::OpBuilder &builder, const c10::TypePtr &c10Type,
                    mlir::Type torchType, mlir::Value loaded,
                    mlir::ModuleOp moduleOp, mlir::Location loc) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

  switch (c10Type->kind()) {
  case c10::TypeKind::TensorType: {
    // Tensor: i64 → AtenTensorHandle → TVMFFIObjectHandle.
    // The dispatcher returns an AtenTensorHandle in the StableIValue slot;
    // we must pack it back into a TVMFFIObjectHandle for the LLVM pipeline.

    // Convert i64 back to AtenTensorHandle pointer.
    mlir::Value atenHandle =
        mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, loaded);

    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> packFn =
        libtriton::conversion::utils::getOrCreateTensorToTVMFFIObject(moduleOp);
    if (mlir::failed(packFn)) {
      return mlir::failure();
    }

    // Allocate stack slot for TVMFFIObjectHandle output.
    mlir::Value outSlot = mlir::LLVM::AllocaOp::create(
        builder, loc, ptrTy, ptrTy,
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 1));

    // Call mLibTritonTensorToTVMFFIObject(atenHandle, &outSlot).
    mlir::LLVM::CallOp::create(builder, loc, *packFn,
                               mlir::ValueRange{atenHandle, outSlot});

    // Load and return the TVMFFIObjectHandle.
    return mlir::LLVM::LoadOp::create(builder, loc, ptrTy, outSlot).getResult();
  }
  case c10::TypeKind::BoolType: {
    // Bool: trunc i64 to i1.
    return mlir::LLVM::TruncOp::create(builder, loc,
                                       mlir::IntegerType::get(ctx, 1), loaded)
        .getResult();
  }
  case c10::TypeKind::IntType:
  case c10::TypeKind::SymIntType: {
    // Int/SymInt: already i64, pass through.
    return loaded;
  }
  case c10::TypeKind::FloatType:
  case c10::TypeKind::SymFloatType: {
    // Float: bitcast i64 to f64.
    return mlir::LLVM::BitcastOp::create(builder, loc,
                                         mlir::Float64Type::get(ctx), loaded)
        .getResult();
  }
  case c10::TypeKind::NoneType: {
    // None: no value produced.
    return mlir::Value();
  }
  case c10::TypeKind::OptionalType: {
    // TODO: support Optional result type unpacking with CondBr blocks,
    // matching the boxed encoding produced by buildStableIValue.
    return mlir::failure();
  }
  case c10::TypeKind::DeviceObjType: {
    // Device: unpack i64 → llvm.struct<(i32, i32)>.
    // PyTorch StableIValue format: (device_type << 32) | device_index
    // Lower 32 bits = device_index, upper 32 bits = device_type.
    mlir::Value devIdx =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, loaded);
    mlir::Value shift = mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, 32);
    mlir::Value shifted =
        mlir::LLVM::LShrOp::create(builder, loc, loaded, shift);
    mlir::Value devType =
        mlir::LLVM::TruncOp::create(builder, loc, i32Ty, shifted);

    mlir::LLVM::LLVMStructType structTy =
        mlir::LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty});
    mlir::Value undef = mlir::LLVM::UndefOp::create(builder, loc, structTy);
    mlir::Value s0 =
        mlir::LLVM::InsertValueOp::create(builder, loc, undef, devType,
                                          llvm::ArrayRef<int64_t>{0})
            .getResult();
    return mlir::LLVM::InsertValueOp::create(builder, loc, s0, devIdx,
                                             llvm::ArrayRef<int64_t>{1})
        .getResult();
  }
  case c10::TypeKind::ListType: {
    // TODO: support List return type resolution.
    return mlir::failure();
  }
  default:
    return mlir::failure();
  }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public C API entry point
//===----------------------------------------------------------------------===//

int mLibTritonSchemaDispatchTorchAtenOp(
    MlirOperation op, MlirValue *operands, MlirValue *results,
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
      mlir::Type torchType = mlirOp->getOperand(i).getType();
      mlir::FailureOr<mlir::Value> built = buildStableIValue(
          mlirRewriter, argType, torchType, adaptedVal, moduleOp, loc);
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
      libtriton::conversion::utils::getOrCreateaoti_torch_call_dispatcher(
          moduleOp);
  if (mlir::failed(calleeOrErr)) {
    return 1;
  }

  // Create global string constants for op_name and overload_name.
  mlir::Value opNamePtr = libtriton::conversion::utils::getOrCreateGlobalString(
      mlirRewriter, loc, moduleOp, "op", dispatcherOpName);
  mlir::Value overloadNamePtr =
      libtriton::conversion::utils::getOrCreateGlobalString(
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
      mlir::Type torchResultType = mlirOp->getResult(i).getType();
      mlir::FailureOr<mlir::Value> resolved = resolveStableIValue(
          mlirRewriter, retType, torchResultType, loaded, moduleOp, loc);
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
