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

/// Helper: given a TVMFFIAny* slot, load the TVMFFIObjectHandle from slot[2],
/// inttoptr it, and advance past the 24-byte header to produce a DLTensor*.
static mlir::Value getDLTensorPtr(mlir::OpBuilder &builder, mlir::Value slot) {
  mlir::Location loc = slot.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType anyTy =
      libtriton::conversion::utils::getTVMFFIAnyType(ctx);
  mlir::IntegerType i8Ty = mlir::IntegerType::get(ctx, 8);

  mlir::Value handlePtr =
      mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
  mlir::Value handleInt =
      mlir::LLVM::LoadOp::create(builder, loc, i64Ty, handlePtr);
  mlir::Value handleObj =
      mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, handleInt);

  return mlir::LLVM::GEPOp::create(
      builder, loc, ptrTy, i8Ty, handleObj,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
}

/// Packs an LLVM value into a TVMFFIAny struct for the FFI return slot.
static mlir::FailureOr<mlir::Value>
To(mlir::Type type, mlir::OpBuilder &builder, mlir::Value input) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::LLVM::LLVMStructType anyTy =
      libtriton::conversion::utils::getTVMFFIAnyType(ctx);

  if (mlir::isa<mlir::torch::Torch::BaseTensorType>(type)) {
    return input;
  } else if (mlir::isa<mlir::torch::Torch::BoolType>(type)) {
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIBool);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0});
    mlir::Value pl = mlir::LLVM::ZExtOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 64), input);
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, pl,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::IntType>(type)) {
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIInt);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0});
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, input,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::FloatType>(type)) {
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFIFloat);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0});
    mlir::Value pl = mlir::LLVM::BitcastOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 64), input);
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, pl,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::NoneType>(type)) {
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32), kTVMFFINone);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0});
    mlir::Value zero = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 64), 0);
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, zero,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::ListType>(type)) {
    // List types stored as kTVMFFIArray=71; return the pointer directly
    // (handler wraps in TVMFFIAny at the call site, like tensor).
    return input;
  } else if (mlir::isa<mlir::torch::Torch::DeviceType>(type)) {
    // Device: pack struct<(i32, i32)> into combined i64 with
    // (device_index << 32) | device_type, stored as kTVMFFIDevice=6.
    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, input.getLoc(), anyTy);
    mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 32),
        kTVMFFIDevice);
    result = mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                               result, typeIdx,
                                               llvm::ArrayRef<int64_t>{0});
    // Extract device_type (field 0) and device_index (field 1).
    mlir::Value deviceType = mlir::LLVM::ExtractValueOp::create(
        builder, input.getLoc(), input, llvm::ArrayRef<int64_t>{0});
    mlir::Value deviceIndex = mlir::LLVM::ExtractValueOp::create(
        builder, input.getLoc(), input, llvm::ArrayRef<int64_t>{1});
    // Combine: (device_index << 32) | device_type.
    mlir::Value devType64 = mlir::LLVM::ZExtOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 64), deviceType);
    mlir::Value devIdx64 = mlir::LLVM::ZExtOp::create(
        builder, input.getLoc(), mlir::IntegerType::get(ctx, 64), deviceIndex);
    mlir::Value shifted = mlir::LLVM::ShlOp::create(
        builder, input.getLoc(), devIdx64,
        mlir::LLVM::ConstantOp::create(builder, input.getLoc(),
                                       mlir::IntegerType::get(ctx, 64), 32));
    mlir::Value combined =
        mlir::LLVM::OrOp::create(builder, input.getLoc(), devType64, shifted);
    return mlir::LLVM::InsertValueOp::create(builder, input.getLoc(), anyTy,
                                             result, combined,
                                             llvm::ArrayRef<int64_t>{2})
        .getResult();
  } else {
    return mlir::failure();
  }
}

/// Unpacks a TVMFFIAny struct from the FFI argument slot.
static mlir::FailureOr<mlir::Value>
From(mlir::Type type, mlir::OpBuilder &builder, mlir::Value ptr) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::LLVM::LLVMStructType anyTy =
      libtriton::conversion::utils::getTVMFFIAnyType(ctx);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);

  if (mlir::isa<mlir::torch::Torch::BaseTensorType>(type)) {
    mlir::Value vobj =
        mlir::LLVM::GEPOp::create(builder, ptr.getLoc(), ptrTy, anyTy, ptr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value i64 = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), vobj);
    return mlir::LLVM::IntToPtrOp::create(builder, ptr.getLoc(), ptrTy, i64)
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::BoolType>(type)) {
    mlir::Value payloadPtr =
        mlir::LLVM::GEPOp::create(builder, ptr.getLoc(), ptrTy, anyTy, ptr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), payloadPtr);
    return mlir::LLVM::TruncOp::create(builder, loaded.getLoc(),
                                       mlir::IntegerType::get(ctx, 1), loaded)
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::IntType>(type)) {
    mlir::Value payloadPtr =
        mlir::LLVM::GEPOp::create(builder, ptr.getLoc(), ptrTy, anyTy, ptr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    return mlir::LLVM::LoadOp::create(builder, ptr.getLoc(),
                                      mlir::IntegerType::get(ctx, 64),
                                      payloadPtr)
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::FloatType>(type)) {
    mlir::Value payloadPtr =
        mlir::LLVM::GEPOp::create(builder, ptr.getLoc(), ptrTy, anyTy, ptr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), payloadPtr);
    return mlir::LLVM::BitcastOp::create(builder, loaded.getLoc(),
                                         mlir::Float64Type::get(ctx), loaded)
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::NoneType>(type)) {
    return mlir::Value();
  } else if (mlir::isa<mlir::torch::Torch::ListType>(type)) {
    // List types (e.g., !torch.list<int>) are stored as kTVMFFIArray=71
    // with v_obj pointing to the TVMFFIObjectHandle.
    mlir::Value vobj =
        mlir::LLVM::GEPOp::create(builder, ptr.getLoc(), ptrTy, anyTy, ptr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value i64 = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), vobj);
    return mlir::LLVM::IntToPtrOp::create(builder, ptr.getLoc(), ptrTy, i64)
        .getResult();
  } else if (mlir::isa<mlir::torch::Torch::DeviceType>(type)) {
    // Device is stored as kTVMFFIDevice=6 with combined
    // (device_index << 32) | device_type in the i64 payload.
    mlir::Value payloadPtr =
        mlir::LLVM::GEPOp::create(builder, ptr.getLoc(), ptrTy, anyTy, ptr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value loaded = mlir::LLVM::LoadOp::create(
        builder, ptr.getLoc(), mlir::IntegerType::get(ctx, 64), payloadPtr);
    // Extract device_type (low 32 bits) and device_index (high 32 bits).
    mlir::Value deviceType = mlir::LLVM::TruncOp::create(
        builder, loaded.getLoc(), mlir::IntegerType::get(ctx, 32), loaded);
    mlir::Value shifted = mlir::LLVM::LShrOp::create(
        builder, loaded.getLoc(), loaded,
        mlir::LLVM::ConstantOp::create(builder, loaded.getLoc(),
                                       mlir::IntegerType::get(ctx, 64), 32));
    mlir::Value deviceIndex = mlir::LLVM::TruncOp::create(
        builder, loaded.getLoc(), mlir::IntegerType::get(ctx, 32), shifted);
    // Build struct<(i32, i32)>.
    mlir::LLVM::LLVMStructType deviceTy =
        mlir::LLVM::LLVMStructType::getLiteral(
            ctx,
            {mlir::IntegerType::get(ctx, 32), mlir::IntegerType::get(ctx, 32)});
    mlir::Value undef =
        mlir::LLVM::UndefOp::create(builder, ptr.getLoc(), deviceTy);
    mlir::Value withType = mlir::LLVM::InsertValueOp::create(
        builder, ptr.getLoc(), deviceTy, undef, deviceType,
        llvm::ArrayRef<int64_t>{0});
    return mlir::LLVM::InsertValueOp::create(builder, ptr.getLoc(), deviceTy,
                                             withType, deviceIndex,
                                             llvm::ArrayRef<int64_t>{1})
        .getResult();
  } else {
    return mlir::failure();
  }
}

//===----------------------------------------------------------------------===//
// Guard checks — dispatched on tvm_ffi.guard argument attributes
//===----------------------------------------------------------------------===//

static mlir::Value buildGuards(mlir::OpBuilder &builder, mlir::Value slot,
                               mlir::Attribute attr) {
  mlir::Location loc = slot.getLoc();
  mlir::MLIRContext *ctx = builder.getContext();

  mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
  mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::LLVM::LLVMStructType dlTensorTy =
      conversion::utils::getDLTensorType(ctx);
  mlir::LLVM::LLVMStructType anyTy =
      libtriton::conversion::utils::getTVMFFIAnyType(ctx);

  if (CudaDeviceGuardAttr cudaGuard =
          mlir::dyn_cast<tvm_ffi::CudaDeviceGuardAttr>(attr)) {
    int64_t deviceType = cudaGuard.getDeviceType();
    int64_t deviceIndex = cudaGuard.getDeviceIndex();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value typeGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1, 0});
    mlir::Value loadedType =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeGep);

    mlir::Value idGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1, 1});
    mlir::Value loadedId =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, idGep);

    mlir::Value expectedType =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, deviceType);
    mlir::Value typeCmp = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loadedType, expectedType);

    mlir::Value expectedId =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, deviceIndex);
    mlir::Value idCmp = mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loadedId, expectedId);

    return mlir::LLVM::AndOp::create(builder, loc, typeCmp, idCmp);
  } else if (DimensionGuardAttr dimGuard =
                 mlir::dyn_cast<DimensionGuardAttr>(attr)) {
    int64_t expectedVal = dimGuard.getExpected();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value ndimGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value ndimVal =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, ndimGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, ndimVal, expected);
  } else if (mlir::isa<DtypeGuardAttr>(attr)) {
    // TODO: The DtypeGuardAttr has no parameters yet. Once dtype fields
    // (code/bits/lanes) are added, compare DLDataType from DLTensor field 3.
    // For now, read the DLDataType struct as a placeholder.
    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value dtypeGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 3});
    mlir::LLVM::LLVMStructType dlDtypeTy =
        conversion::utils::getDLDataType(ctx);
    mlir::LLVM::LoadOp::create(builder, loc, dlDtypeTy, dtypeGep);

    return mlir::LLVM::ConstantOp::create(builder, loc,
                                          mlir::IntegerType::get(ctx, 1), 1);
  } else if (SizeGuardAttr sizeGuard = mlir::dyn_cast<SizeGuardAttr>(attr)) {
    int64_t index = sizeGuard.getIndex();
    int64_t expectedVal = sizeGuard.getExpected();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value shapePtrGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 4});
    mlir::Value shapePtr =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, shapePtrGep);

    mlir::Value elemGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, i64Ty, shapePtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
    mlir::Value elemVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, elemGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, elemVal, expected);
  } else if (StorageOffsetGuardAttr offsetGuard =
                 mlir::dyn_cast<StorageOffsetGuardAttr>(attr)) {
    int64_t expectedVal = offsetGuard.getExpected();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value offsetGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 6});
    mlir::Value offsetVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, offsetGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, offsetVal, expected);
  } else if (StrideGuardAttr strideGuard =
                 mlir::dyn_cast<StrideGuardAttr>(attr)) {
    int64_t index = strideGuard.getIndex();
    int64_t expectedVal = strideGuard.getExpected();

    mlir::Value dlTensorPtr = getDLTensorPtr(builder, slot);

    mlir::Value stridePtrGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, dlTensorTy, dlTensorPtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 5});
    mlir::Value stridePtr =
        mlir::LLVM::LoadOp::create(builder, loc, ptrTy, stridePtrGep);

    mlir::Value elemGep =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, i64Ty, stridePtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
    mlir::Value elemVal =
        mlir::LLVM::LoadOp::create(builder, loc, i64Ty, elemGep);

    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i64Ty, expectedVal);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, elemVal, expected);
  } else if (mlir::isa<TensorTypeGuardAttr>(attr)) {
    mlir::Value typeCodePtr =
        mlir::LLVM::GEPOp::create(builder, loc, ptrTy, anyTy, slot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
    mlir::Value loadedTypeCode =
        mlir::LLVM::LoadOp::create(builder, loc, i32Ty, typeCodePtr);
    mlir::Value expected =
        mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, kTVMFFITensor);
    return mlir::LLVM::ICmpOp::create(
        builder, loc, mlir::LLVM::ICmpPredicate::eq, loadedTypeCode, expected);
  }
  return {};
}

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
    mlir::LLVM::LLVMStructType anyTy =
        libtriton::conversion::utils::getTVMFFIAnyType(context);
    for (auto [i, arg] : llvm::enumerate(op.getArguments())) {
      mlir::Value slot =
          mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, argsPtr,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{i});
      mlir::Type argTy = arg.getType();
      mlir::FailureOr<mlir::Value> loaded = From(argTy, rewriter, slot);
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
          mlir::Value guardResult = buildGuards(rewriter, slot, g);
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
                To(operandTy, rewriter, casted);
            if (mlir::failed(result)) {
              return op.emitError("unsupported return type: ") << operandTy;
            }
            // For tensor/list types, the handler returns a TVMFFIObjectHandle
            // (!llvm.ptr). We need to wrap it into a TVMFFIAny struct before
            // storing to retPtr.
            if (mlir::isa<mlir::torch::Torch::BaseTensorType>(operandTy) ||
                mlir::isa<mlir::torch::Torch::ListType>(operandTy)) {
              mlir::Value handle = *result;
              mlir::Value wrapped =
                  mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
              int32_t typeCode =
                  mlir::isa<mlir::torch::Torch::ListType>(operandTy)
                      ? kTVMFFIArray
                      : kTVMFFITensor;
              mlir::Value typeIdx = mlir::LLVM::ConstantOp::create(
                  rewriter, loc, i32Ty, typeCode);
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
              mlir::LLVM::StoreOp::create(rewriter, loc, *result, retPtr);
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
