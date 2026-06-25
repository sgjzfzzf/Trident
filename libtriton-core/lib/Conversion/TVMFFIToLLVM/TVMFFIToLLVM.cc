//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "libtriton-core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "libtriton-core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/GlobalString.h"
#include "libtriton-core/Conversion/Utils/LibTritonCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/StdLibCAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "libtriton-core/Conversion/Utils/Type.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIAttributes.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

#include <string>

namespace libtriton::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

struct Aux {};

static mlir::LLVM::LLVMStructType getTVMFFIAnyType(mlir::MLIRContext *context) {
  mlir::IntegerType i32Ty = mlir::IntegerType::get(context, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(context, 64);
  return mlir::LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i64Ty});
}

/// Helper: given a TVMFFIAny* slot, load the TVMFFIObjectHandle from slot[2],
/// inttoptr it, and advance past the 24-byte header to produce a DLTensor*.
static mlir::Value getDLTensorPtr(mlir::OpBuilder &builder, mlir::Value slot) {
  mlir::Location loc = slot.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
  mlir::IntegerType i8Ty = mlir::IntegerType::get(ctx, 8);

  mlir::Value handlePtr =
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
  mlir::Value handleInt =
      mlir::LLVM::LoadOp::create(builder, loc, i64Ty, handlePtr);
  mlir::Value handleObj =
      mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, handleInt);

  return mlir::LLVM::GEPOp::create(
      builder, loc, ptrTy, i8Ty, handleObj,
      mlir::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
}

//===----------------------------------------------------------------------===//
// Type conversion handlers for packing/unpacking TVM FFI arguments
//===----------------------------------------------------------------------===//

/// CRTP base: auto-generates matches(mlir::Type) from the Torch type parameter.
template <typename TorchType> struct TypeHandlerBase {
  static bool matches(mlir::Type type) { return mlir::isa<TorchType>(type); }
};

struct BaseTensorHandler : TypeHandlerBase<mlir::torch::Torch::BaseTensorType> {
  /// The input is already a TVMFFIObjectHandle (!llvm.ptr) from the function
  /// body (via TorchExtToLLVM lowering which keeps values as handles).
  /// Just return it directly — ConvertFuncOp wraps it into a TVMFFIAny struct.
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    return input;
  }

  /// Extracts the TVMFFIObjectHandle (v_obj) from the TVMFFIAny struct and
  /// returns it as a !llvm.ptr. No conversion to AtenTensorHandle here;
  /// that happens when the handle reaches an ATen op.
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &builder,
                                           mlir::Value ptr, Aux &aux) {
    mlir::Location loc = ptr.getLoc();
    mlir::MLIRContext *context = builder.getContext();
    mlir::LLVM::LLVMPointerType ptrTy =
        mlir::LLVM::LLVMPointerType::get(context);

    // Extract the TVMFFIObjectHandle (v_obj) from the TVMFFIAny struct.
    // v_obj is at field index 2 in the {i32, i32, i64} struct.
    mlir::Value vobj = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, getTVMFFIAnyType(context), ptr,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value i64 = mlir::LLVM::LoadOp::create(
        builder, loc, mlir::IntegerType::get(context, 64), vobj);
    return mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, i64).getResult();
  }
};

struct BoolHandler : TypeHandlerBase<mlir::torch::Torch::BoolType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIBool);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0})
                 .getResult();
    mlir::Value pl =
        mlir::LLVM::ZExtOp::create(builder, input.getLoc(),
                                   mlir::IntegerType::get(ctx, 64), input)
            .getResult();
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, pl,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &builder,
                                           mlir::Value ptr, Aux &) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Value payloadPtr = mlir::LLVM::GEPOp::create(
        builder, ptr.getLoc(), mlir::LLVM::LLVMPointerType::get(ctx),
        getTVMFFIAnyType(ctx), ptr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), payloadPtr);
    return mlir::LLVM::TruncOp::create(
               builder, loaded.getLoc(),
               mlir::IntegerType::get(builder.getContext(), 1), loaded)
        .getResult();
  }
};

struct IntHandler : TypeHandlerBase<mlir::torch::Torch::IntType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIInt);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0})
                 .getResult();
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, input,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &builder,
                                           mlir::Value ptr, Aux &) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Value payloadPtr = mlir::LLVM::GEPOp::create(
        builder, ptr.getLoc(), mlir::LLVM::LLVMPointerType::get(ctx),
        getTVMFFIAnyType(ctx), ptr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    return mlir::LLVM::LoadOp::create(builder, ptr.getLoc(),
                                      mlir::IntegerType::get(ctx, 64),
                                      payloadPtr)
        .getResult();
  }
};

struct FloatHandler : TypeHandlerBase<mlir::torch::Torch::FloatType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIFloat);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0})
                 .getResult();
    mlir::Value pl =
        mlir::LLVM::BitcastOp::create(builder, input.getLoc(),
                                      mlir::IntegerType::get(ctx, 64), input)
            .getResult();
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, pl,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &builder,
                                           mlir::Value ptr, Aux &) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::Value payloadPtr = mlir::LLVM::GEPOp::create(
        builder, ptr.getLoc(), mlir::LLVM::LLVMPointerType::get(ctx),
        getTVMFFIAnyType(ctx), ptr, mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), payloadPtr);
    return mlir::LLVM::BitcastOp::create(
               builder, loaded.getLoc(),
               mlir::Float64Type::get(builder.getContext()), loaded)
        .getResult();
  }
};

struct NoneHandler : TypeHandlerBase<mlir::torch::Torch::NoneType> {
  static mlir::FailureOr<mlir::Value> to(mlir::OpBuilder &builder,
                                         mlir::Value input) {
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(ctx);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFINone);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0})
                 .getResult();
    mlir::Value zero = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 64), 0);
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, zero,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  }
  static mlir::FailureOr<mlir::Value> from(mlir::OpBuilder &, mlir::Value,
                                           Aux &) {
    return mlir::Value();
  }
};

//===----------------------------------------------------------------------===//
// Guard handler framework — dispatches on tvm_ffi.guard argument attributes
//===----------------------------------------------------------------------===//

/// Base CRTP class for guard handlers. Subclasses must define:
///   static ::mlir::Value
///   check(::mlir::OpBuilder &builder, ::mlir::Value slot,
///         ::mlir::Attribute attr);
template <typename Concrete, typename GuardAttr> struct GuardHandlerBase {
  using AttrType = GuardAttr;

  static bool matches(mlir::Attribute attr) {
    return mlir::isa<GuardAttr>(attr);
  }

  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value, AttrType) {
    return mlir::LLVM::ConstantOp::create(
        builder, builder.getUnknownLoc(),
        mlir::IntegerType::get(builder.getContext(), 1), 1);
  }
};

// ── Concrete guard handlers (placeholders — real checks TBD) ──

/// Helper: load type_code from slot[0] and compare against expected value.
static mlir::Value checkTypeCode(mlir::OpBuilder &builder, mlir::Value slot,
                                 int32_t expectedTypeCode) {
  mlir::Location loc = slot.getLoc();
  mlir::IntegerType i32Ty = mlir::IntegerType::get(builder.getContext(), 32);
  mlir::LLVM::LLVMPointerType ptrTy =
      mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(builder.getContext());

  mlir::Value typeCodePtr =
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
  mlir::Value loadedTypeCode =
      mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeCodePtr);
  mlir::Value expected =
      mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, expectedTypeCode);
  return mlir::LLVM::ICmpOp::create(builder, loc, mlir::LLVM::ICmpPredicate::eq,
                                    loadedTypeCode, expected);
}

struct CUDADeviceGuardHandler
    : GuardHandlerBase<CUDADeviceGuardHandler, tvm_ffi::CudaDeviceGuardAttr> {
  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value slot,
                           AttrType attr) {
    int64_t deviceType = attr.getDeviceType();
    int64_t deviceIndex = attr.getDeviceIndex();

    mlir::Location loc = slot.getLoc();
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType dlTensorTy =
        conversion::utils::getDLTensorType(ctx);

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value typeGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 1, 0});
    mlir::Value loadedType =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeGep);

    mlir::Value idGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 1, 1});
    mlir::Value loadedId =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, idGep);

    mlir::Value expectedType = mlir::LLVM::ConstantOp::create(
        builder, loc, i32Ty, static_cast<int32_t>(deviceType));
    mlir::Value typeCmp = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loadedType, expectedType);

    mlir::Value expectedId = mlir::LLVM::ConstantOp::create(
        builder, loc, i32Ty, static_cast<int32_t>(deviceIndex));
    mlir::Value idCmp = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loadedId, expectedId);

    return mlir::LLVM::AndOp::create(builder, loc, typeCmp, idCmp);
  }
};

struct DimensionGuardHandler
    : GuardHandlerBase<DimensionGuardHandler, DimensionGuardAttr> {
  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value slot,
                           AttrType attr) {
    int64_t expectedVal = attr.getExpected();

    mlir::Location loc = slot.getLoc();
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType dlTensorTy =
        conversion::utils::getDLTensorType(ctx);

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value ndimGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value ndimVal =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, ndimGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, ndimVal, expected);
  }
};

struct DTypeGuardHandler : GuardHandlerBase<DTypeGuardHandler, DtypeGuardAttr> {
  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value slot,
                           AttrType) {
    // TODO: The DtypeGuardAttr has no parameters yet. Once dtype fields
    // (code/bits/lanes) are added, compare DLDataType from DLTensor field 3.
    // For now, read the DLDataType struct as a placeholder.
    mlir::Location loc = slot.getLoc();
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType dlTensorTy =
        conversion::utils::getDLTensorType(ctx);

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    // Load DLDataType from DLTensor field 3 (code, bits, lanes).
    mlir::Value dtypeGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 3});
    mlir::LLVM::LLVMStructType dlDtypeTy =
        conversion::utils::getDLDataType(ctx);
    mlir::LLVM::LoadOp::create(builder, loc, dlDtypeTy, dtypeGep);

    // Always succeed for now (dtype comparison TBD).
    return mlir::LLVM::ConstantOp::create(
        builder, loc, mlir::IntegerType::get(builder.getContext(), 1), 1);
  }
};

struct SizeGuardHandler : GuardHandlerBase<SizeGuardHandler, SizeGuardAttr> {
  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value slot,
                           AttrType attr) {
    int64_t index = attr.getIndex();
    int64_t expectedVal = attr.getExpected();

    mlir::Location loc = slot.getLoc();
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType dlTensorTy =
        conversion::utils::getDLTensorType(ctx);

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value shapePtrGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 4});
    mlir::Value shapePtr =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, shapePtrGep);

    mlir::Value elemGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, shapePtr,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
    mlir::Value elemVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, elemGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, elemVal, expected);
  }
};

struct StorageOffsetGuardHandler
    : GuardHandlerBase<StorageOffsetGuardHandler, StorageOffsetGuardAttr> {
  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value slot,
                           AttrType attr) {
    int64_t expectedVal = attr.getExpected();

    mlir::Location loc = slot.getLoc();
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType dlTensorTy =
        conversion::utils::getDLTensorType(ctx);

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value offsetGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 6});
    mlir::Value offsetVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, offsetGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, offsetVal, expected);
  }
};

struct StrideGuardHandler
    : GuardHandlerBase<StrideGuardHandler, StrideGuardAttr> {
  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value slot,
                           AttrType attr) {
    int64_t index = attr.getIndex();
    int64_t expectedVal = attr.getExpected();

    mlir::Location loc = slot.getLoc();
    mlir::MLIRContext *ctx = builder.getContext();
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType dlTensorTy =
        conversion::utils::getDLTensorType(ctx);

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value stridePtrGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  mlir::ArrayRef<mlir::LLVM::GEPArg>{0, 5});
    mlir::Value stridePtr =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, stridePtrGep);

    mlir::Value elemGep = mlir::LLVM::GEPOp::create(
        builder, loc, ptrTy, i64Ty, stridePtr,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
    mlir::Value elemVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, elemGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, elemVal, expected);
  }
};

struct TensorTypeGuardHandler
    : GuardHandlerBase<TensorTypeGuardHandler, TensorTypeGuardAttr> {
  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value slot,
                           AttrType) {
    return checkTypeCode(builder, slot, kTVMFFITensor);
  }
};

/// Fold-expression dispatch over variadic handler types.
template <typename... Handlers> struct GuardDispatch {
  static mlir::Value check(mlir::OpBuilder &builder, mlir::Value slot,
                           mlir::Attribute attr) {
    mlir::Value result;
    // Try each handler; the first one that matches wins.
    (void)((Handlers::matches(attr)
                ? (result = Handlers::check(
                       builder, slot,
                       mlir::cast<typename Handlers::AttrType>(attr)),
                   true)
                : false) ||
           ...);
    return result;
  }
};

using AllGuardHandlers =
    GuardDispatch<CUDADeviceGuardHandler, DimensionGuardHandler,
                  DTypeGuardHandler, SizeGuardHandler,
                  StorageOffsetGuardHandler, StrideGuardHandler,
                  TensorTypeGuardHandler>;

//===----------------------------------------------------------------------===//
// Variadic dispatch: folds over handlers, short-circuits on first match
//===----------------------------------------------------------------------===//

// ── Value-based dispatch (all handlers' to() return a value, no ptr) ──
template <typename Handler> struct HandlerCaller {
  static mlir::FailureOr<mlir::Value>
  tryTo(mlir::Type type, mlir::OpBuilder &builder, mlir::Value input) {
    return Handler::matches(type) ? Handler::to(builder, input)
                                  : mlir::FailureOr<mlir::Value>();
  }

  static mlir::FailureOr<mlir::Value> tryFrom(mlir::Type type,
                                              mlir::OpBuilder &builder,
                                              mlir::Value ptr, Aux &aux) {
    return Handler::matches(type) ? Handler::from(builder, ptr, aux)
                                  : mlir::FailureOr<mlir::Value>();
  }
};

template <typename... Handlers> struct TypeDispatch {
  static mlir::FailureOr<mlir::Value>
  to(mlir::Type type, mlir::OpBuilder &builder, mlir::Value input) {
    mlir::FailureOr<mlir::Value> result = mlir::failure();
    (mlir::succeeded(
         result = HandlerCaller<Handlers>::tryTo(type, builder, input)) ||
     ...);
    return result;
  }

  static mlir::FailureOr<mlir::Value>
  from(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr, Aux &aux) {
    mlir::FailureOr<mlir::Value> result = mlir::failure();
    (mlir::succeeded(
         result = HandlerCaller<Handlers>::tryFrom(type, builder, ptr, aux)) ||
     ...);
    return result;
  }
};

using AllHandlers = TypeDispatch<BaseTensorHandler, BoolHandler, IntHandler,
                                 FloatHandler, NoneHandler>;

/// Converts tvm_ffi.func to func.func by converting the function type
/// signature through the type converter and inlining the region body.
class ConvertFuncOp : public mlir::OpConversionPattern<FuncOp> {
public:
  ConvertFuncOp(mlir::LLVMTypeConverter &typeConverter,
                mlir::MLIRContext *context)
      : mlir::OpConversionPattern<FuncOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *context = rewriter.getContext();

    // Without support for tvm_ffi.Array, at most one return value is allowed.
    assert(op.getFunctionType().getResults().size() <= 1 &&
           "Without the support of `tvm.ffi.Array`, we only support at most "
           "one return value temporarily.");

    // TVM-FFI C ABI: int32_t(void*, void*, int32_t, void*)
    mlir::IntegerType i32Ty = rewriter.getIntegerType(32);
    mlir::LLVM::LLVMPointerType ptrTy =
        mlir::LLVM::LLVMPointerType::get(context);
    mlir::LLVM::LLVMFunctionType funcType =
        mlir::LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, i32Ty, ptrTy});

    // Create the LLVM function, then build all blocks from scratch.
    std::string tvmffiFuncName =
        llvm::formatv("__tvm_ffi_{0}", op.getSymName());
    mlir::LLVM::LLVMFuncOp llvmFunc = mlir::LLVM::LLVMFuncOp::create(
        rewriter, loc, tvmffiFuncName, funcType, mlir::LLVM::Linkage::External);
    mlir::Region &region = llvmFunc.getBody();
    mlir::IRMapping mapping;

    // Step 1: Create the entry block with ABI arguments from the function type.
    // TVM-FFI C ABI: int32_t(void* handle, TVMFFIAny* args, int32_t num_args,
    // TVMFFIAny* result)
    mlir::Block *entryBlock = llvmFunc.addEntryBlock(rewriter);
    mlir::Value argsPtr = entryBlock->getArgument(1); // TVMFFIAny* args
    mlir::Value retPtr = entryBlock->getArgument(3);  // TVMFFIAny* result

    // Step 2: Parse arguments. Type checks may split the entry block,
    // producing error/continue blocks that naturally precede the body.
    rewriter.setInsertionPointToStart(entryBlock);
    mlir::LLVM::LLVMStructType anyTy = getTVMFFIAnyType(context);
    Aux aux;
    for (auto [i, arg] : llvm::enumerate(op.getArguments())) {
      mlir::Value slot =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, argsPtr,
                                    mlir::ArrayRef<mlir::LLVM::GEPArg>{i});
      mlir::Type argTy = arg.getType();
      mlir::FailureOr<mlir::Value> loaded =
          AllHandlers::from(argTy, rewriter, slot, aux);
      if (mlir::failed(loaded)) {
        return op.emitError("unsupported input type: ") << argTy;
      }
      mlir::Value casted = mlir::UnrealizedConversionCastOp::create(
                               rewriter, loc, argTy, *loaded)
                               .getResult(0);
      mapping.map(arg, casted);

      // ── Guard checking: if any guard fails, return immediately with -1 ──
      if (mlir::ArrayAttr guardAttrs = mlir::dyn_cast_or_null<mlir::ArrayAttr>(
              op.getArgAttr(i, "tvm_ffi.guard"))) {
        for (mlir::Attribute g : guardAttrs) {
          mlir::Value guardResult = AllGuardHandlers::check(rewriter, slot, g);
          if (!guardResult) {
            return op.emitError("unsupported guard attribute on argument ")
                   << i;
          }
          mlir::Block *currentBlock = rewriter.getInsertionBlock();
          mlir::Block *failBlock = rewriter.createBlock(&region);
          rewriter.setInsertionPointToStart(failBlock);
          // Set the TVM FFI error kind to GuardMatchException before
          // returning -1, so the caller can identify the failure reason.
          mlir::ModuleOp moduleOp =
              op->template getParentOfType<mlir::ModuleOp>();
          if (!moduleOp) {
            return op.emitError("failed to get parent ModuleOp for guard "
                                "failure error reporting");
          }
          mlir::FailureOr<mlir::LLVM::LLVMFuncOp> errorFn =
              conversion::utils::getOrCreateTVMFFIErrorSetRaisedFromCStr(
                  moduleOp);
          if (mlir::failed(errorFn)) {
            return op.emitError(
                "failed to get or create TVMFFIErrorSetRaisedFromCStr");
          }
          mlir::Value kindPtr = conversion::utils::getOrCreateGlobalString(
              rewriter, loc, moduleOp, "GuardMatchExceptionKind",
              "GuardMatchException");
          std::string errMsg =
              llvm::formatv("argument {0} fails guard check", i);
          mlir::Value msgPtr = conversion::utils::getOrCreateGlobalString(
              rewriter, loc, moduleOp, "GuardMatchExceptionMsg", errMsg);
          mlir::LLVM::CallOp::create(rewriter, loc, *errorFn,
                                     mlir::ValueRange{kindPtr, msgPtr});
          mlir::LLVM::ConstantOp errCode =
              mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, -1);
          mlir::LLVM::ReturnOp::create(rewriter, loc, errCode);

          mlir::Block *contBlock = rewriter.createBlock(&region);
          rewriter.setInsertionPointToEnd(currentBlock);
          mlir::LLVM::CondBrOp::create(rewriter, loc, guardResult, contBlock,
                                       failBlock);

          rewriter.setInsertionPointToStart(contBlock);
        }
      }
    }

    // Step 3: Create mapped body blocks (placed after any split blocks).
    mlir::Block *firstBodyBlock = nullptr;
    for (mlir::Block &blk : op.getBody()) {
      mlir::ConversionPatternRewriter::InsertionGuard guard(rewriter);
      mlir::Block *newBlock = rewriter.createBlock(&region);
      if (firstBodyBlock == nullptr) {
        firstBodyBlock = newBlock;
      }
      mapping.map(&blk, newBlock);
    }

    // Step 4: Branch from the last guard-check block to the first body block.
    assert(firstBodyBlock && "expected at least one block in function body");
    mlir::LLVM::BrOp::create(rewriter, loc, firstBodyBlock);

    // Step 6: Clone original function body ops into the mapped body blocks.
    for (mlir::Block &blk : op.getBody()) {
      mlir::Block *dest = mapping.lookupOrDefault(&blk);
      rewriter.setInsertionPointToEnd(dest);
      for (mlir::Operation &operation : llvm::make_early_inc_range(blk)) {
        if (ReturnOp returnOp = mlir::dyn_cast<ReturnOp>(&operation)) {
          for (mlir::Value operand : returnOp.getOperands()) {
            mlir::Value retVal = mapping.lookupOrDefault(operand);
            mlir::Type operandTy = operand.getType();
            mlir::Value casted =
                mlir::UnrealizedConversionCastOp::create(
                    rewriter, loc, getTypeConverter()->convertType(operandTy),
                    retVal)
                    .getResult(0);
            mlir::FailureOr<mlir::Value> result =
                AllHandlers::to(operandTy, rewriter, casted);
            if (mlir::failed(result)) {
              return op.emitError("unsupported return type: ") << operandTy;
            }
            // For tensor types, the handler returns a TVMFFIObjectHandle
            // (!llvm.ptr). We need to wrap it into a TVMFFIAny struct before
            // storing to retPtr.
            if (mlir::isa<mlir::torch::Torch::BaseTensorType>(operandTy)) {
              mlir::Value handle = result.value();
              mlir::Value wrapped =
                  mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
              mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
                  rewriter, loc, i32Ty, kTVMFFITensor);
              wrapped = mlir::LLVM::InsertValueOp::create(
                  rewriter, loc, anyTy, wrapped, typeIdx,
                  llvm::ArrayRef<int64_t>{0});
              mlir::Value vObj = mlir::LLVM::PtrToIntOp::create(
                  rewriter, loc, rewriter.getIntegerType(64), handle);
              wrapped = mlir::LLVM::InsertValueOp::create(rewriter, loc, anyTy,
                                                          wrapped, vObj, {2});
              mlir::LLVM::StoreOp::create(rewriter, loc, wrapped, retPtr);
            } else {
              // POD handlers return a full TVMFFIAny struct; store directly.
              mlir::LLVM::StoreOp::create(rewriter, loc, result.value(),
                                          retPtr);
            }
          }
          mlir::LLVM::ConstantOp cnst =
              mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0);
          mlir::LLVM::ReturnOp::create(rewriter, loc, cnst.getResult());
        } else {
          rewriter.clone(operation, mapping);
        }
      }
    }

    rewriter.replaceOp(op, llvmFunc);
    return mlir::success();
  }
};

class ConvertTVMFFIToLLVMPass
    : public impl::ConvertTVMFFIToLLVMBase<ConvertTVMFFIToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);

    torch::setupBackendTypeConversion(target, typeConverter);
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();
    target.addLegalDialect<mlir::torch::Torch::TorchDialect>();
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct TVMFFIToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    torch::setupBackendTypeConversion(target, typeConverter);
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertFuncOp>(typeConverter, patterns.getContext());
  target.addIllegalDialect<TVMFFIDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::func::FuncDialect,
                         mlir::LLVM::LLVMDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace libtriton::tvm_ffi
