//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/GlobalString.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <ATen/dlpack.h>
#include <c10/core/Device.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch/headeronly/macros/Export.h>
#include <tvm/ffi/c_api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace at {
TORCH_API DLDevice torchDeviceToDLDevice(c10::Device device);
} // namespace at

namespace trident::conversion {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "trident/core/Conversion/Passes.h.inc"

class ConvertArrayLengthOp
    : public mlir::OpConversionPattern<tvm_ffi::ArrayLengthOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ArrayLengthOp op,
                  tvm_ffi::ArrayLengthOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFIABIType::getLLVMType(ctx);
    mlir::Value const one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 1);
    mlir::Value const argument =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getArray(), argument);
    mlir::Value const result =
        TRIDENT_CHECK(conversion::utils::callTVMFFIGlobalFunction(
                          rewriter, loc, module, "ffi.ArraySize",
                          llvm::ArrayRef<mlir::Value>{argument}),
                      return op.emitError("failed to call ffi.ArraySize"));
    mlir::Value const payloadPtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, anyTy, result,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, rewriter.getI64Type(),
                                                    payloadPtr);
    return mlir::success();
  }
};

class ConvertTensorLiteralOp
    : public mlir::OpConversionPattern<tvm_ffi::TensorLiteralOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::TensorLiteralOp op, tvm_ffi::TensorLiteralOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::DenseElementsAttr const dense =
        mlir::cast<mlir::DenseElementsAttr>(op.getValue());
    mlir::RankedTensorType const denseType =
        mlir::cast<mlir::RankedTensorType>(dense.getType());
    mlir::Location loc = op.getLoc();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::LLVM::LLVMPointerType ptrType =
        mlir::LLVM::LLVMPointerType::get(context);
    mlir::IntegerType const i8Type = rewriter.getI8Type();
    mlir::IntegerType i32Type = rewriter.getI32Type();
    mlir::IntegerType const i64Type = rewriter.getI64Type();

    int64_t elementBits;
    int32_t dtypeCode;
    if (mlir::FloatType floatType =
            mlir::dyn_cast<mlir::FloatType>(denseType.getElementType())) {
      elementBits = floatType.getWidth();
      dtypeCode = kDLFloat;
    } else if (mlir::IntegerType const integerType =
                   mlir::dyn_cast<mlir::IntegerType>(
                       denseType.getElementType())) {
      elementBits =
          integerType.isSignlessInteger(1) ? 8 : integerType.getWidth();
      dtypeCode = integerType.isSignlessInteger(1)
                      ? kDLBool
                      : (integerType.isUnsignedInteger() ? kDLUInt : kDLInt);
    } else {
      return op.emitError("unsupported literal element type");
    }

    mlir::Value const dtypeArgument =
        tvm_ffi::DTypeType::build(rewriter, loc, dtypeCode, elementBits);
    mlir::FailureOr<mlir::Value> dtypeResult =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "trident.runtime.tvm_ffi_to_torch_type",
            {dtypeArgument});
    if (mlir::failed(dtypeResult)) {
      return op.emitError("failed to call TVM FFI dtype conversion helper");
    }
    mlir::Value const dtypePayload =
        tvm_ffi::TVMFFIABIType::load(rewriter, loc, dtypeResult.value());
    mlir::Value const dtype =
        mlir::LLVM::TruncOp::create(rewriter, loc, i32Type, dtypePayload);
    llvm::ArrayRef<int64_t> const shape = denseType.getShape();
    mlir::Value const rank = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i64Type, static_cast<int64_t>(shape.size()));
    mlir::Value const shapeCount = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i64Type, static_cast<int64_t>(shape.size()));
    mlir::Value const sizes = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrType, i64Type, shapeCount);
    mlir::Value const strides = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrType, i64Type, shapeCount);
    int64_t stride = 1;
    for (int64_t index = static_cast<int64_t>(shape.size()) - 1; index >= 0;
         --index) {
      mlir::Value const sizeSlot = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrType, i64Type, sizes,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::Value const strideSlot = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrType, i64Type, strides,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::LLVM::StoreOp::create(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, shape[index]),
          sizeSlot);
      mlir::LLVM::StoreOp::create(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, stride),
          strideSlot);
      stride *= shape[index];
    }

    mlir::Type storageType = denseType.getElementType();
    if (mlir::IntegerType const integerType =
            mlir::dyn_cast<mlir::IntegerType>(storageType);
        integerType && integerType.isSignlessInteger(1)) {
      storageType = i8Type;
    }
    mlir::Value const elementCount = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i64Type, dense.getNumElements());
    mlir::Value data = mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType,
                                                    storageType, elementCount);
    auto storeElement = [&](int64_t index, mlir::Attribute value) {
      mlir::Value const slot = mlir::LLVM::GEPOp::create(
          rewriter, loc, ptrType, storageType, data,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::LLVM::StoreOp::create(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, storageType, value),
          slot);
    };
    if (mlir::isa<mlir::FloatType>(storageType)) {
      for (auto [index, value] :
           llvm::enumerate(dense.getValues<llvm::APFloat>())) {
        storeElement(static_cast<int64_t>(index),
                     mlir::FloatAttr::get(storageType, value));
      }
    } else {
      for (auto [index, value] :
           llvm::enumerate(dense.getValues<llvm::APInt>())) {
        if (mlir::cast<mlir::IntegerType>(denseType.getElementType())
                .isSignlessInteger(1)) {
          value = llvm::APInt(8, value.getBoolValue());
        }
        storeElement(static_cast<int64_t>(index),
                     mlir::IntegerAttr::get(storageType, value));
      }
    }

    auto convertDevice =
        [&](int32_t deviceType) -> mlir::FailureOr<mlir::Value> {
      mlir::Value const argument = tvm_ffi::IntType::build(
          rewriter, loc,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i32Type, deviceType));
      mlir::FailureOr<mlir::Value> result =
          conversion::utils::callTVMFFIGlobalFunction(
              rewriter, loc, module,
              "trident.runtime.tvm_ffi_device_to_torch_device_type",
              {argument});
      if (mlir::failed(result)) {
        return mlir::failure();
      }
      mlir::Value const deviceTypePayload =
          tvm_ffi::TVMFFIABIType::load(rewriter, loc, result.value());
      return mlir::LLVM::TruncOp::create(rewriter, loc, i32Type,
                                         deviceTypePayload)
          .getResult();
    };
    mlir::Value const one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, 1);
    mlir::Value const zeroI32 =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Type, 0);
    mlir::Value const zeroI64 =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Type, 0);
    mlir::Value const deviceIndexSlot =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, i32Type, one);
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(
                module)),
        deviceIndexSlot);
    mlir::Value const deviceIndex =
        mlir::LLVM::LoadOp::create(rewriter, loc, i32Type, deviceIndexSlot);
    mlir::Value const cpuOutput =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, one);
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchCreateTensorFromBlob(
                module)),
        mlir::ValueRange{data, rank, sizes, strides, zeroI64, dtype,
                         TRIDENT_CHECK_FAILURE(convertDevice(kDLCPU)), zeroI32,
                         cpuOutput});
    mlir::Value const cpuTensor =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrType, cpuOutput);
    mlir::Value const cudaOutput =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrType, ptrType, one);
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchEmptyStrided(module)),
        mlir::ValueRange{rank, sizes, strides, dtype,
                         TRIDENT_CHECK_FAILURE(convertDevice(kDLCUDA)),
                         deviceIndex, cudaOutput});
    mlir::Value const tensor =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrType, cudaOutput);
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchCopy_(module)),
        mlir::ValueRange{tensor, cpuTensor, zeroI32});
    mlir::LLVM::CallOp::create(
        rewriter, loc,
        TRIDENT_CHECK_FAILURE(
            conversion::utils::getOrCreateAOTITorchDeleteTensorObject(module)),
        cpuTensor);

    mlir::Value tensorArgument = tvm_ffi::IntType::build(rewriter, loc, 0);
    mlir::Value const opaquePtrTypeIndex = mlir::LLVM::ConstantOp::create(
        rewriter, loc, i32Type, kTVMFFIOpaquePtr);
    mlir::LLVM::StoreOp::create(
        rewriter, loc, opaquePtrTypeIndex,
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrType,
                                  tvm_ffi::TVMFFIABIType::getLLVMType(context),
                                  tensorArgument,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
    mlir::Value const opaquePtrPayload =
        mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Type, tensor);
    mlir::LLVM::StoreOp::create(
        rewriter, loc, opaquePtrPayload,
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrType,
                                  tvm_ffi::TVMFFIABIType::getLLVMType(context),
                                  tensorArgument,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
    mlir::FailureOr<mlir::Value> result =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "trident.runtime.tensor_to_tvm_ffi_object",
            {tensorArgument});
    if (mlir::failed(result)) {
      return op.emitError("failed to call TVM FFI tensor conversion helper");
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
        op, tvm_ffi::TVMFFIABIType::getLLVMType(context), result.value());
    return mlir::success();
  }
};

class ConvertCallOp : public mlir::OpConversionPattern<tvm_ffi::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::CallOp op, tvm_ffi::CallOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFIABIType::getLLVMType(ctx);
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::IntegerType const i32Ty = rewriter.getI32Type();
    mlir::IntegerType const i64Ty = rewriter.getI64Type();

    llvm::SmallVector<mlir::Value> arguments(adaptor.getOperands().begin(),
                                             adaptor.getOperands().end());
    mlir::Value const count = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i32Ty, static_cast<int32_t>(arguments.size()));
    mlir::Value const slots = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                       static_cast<int64_t>(arguments.size())));
    for (auto [index, argument] : llvm::enumerate(arguments)) {
      mlir::Value const slot = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), ptrTy, anyTy, slots,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), argument, slot);
    }

    mlir::Value const resultSlot = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 1));
    mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(),
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, 0),
        mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, anyTy,
                                  resultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
    mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(),
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, 0),
        mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, anyTy,
                                  resultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));
    mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(),
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 0),
        mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, anyTy,
                                  resultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));

    std::string const callee = llvm::formatv("__tvm_ffi_{0}", op.getCallee());
    mlir::func::CallOp::create(
        rewriter, op.getLoc(), callee, mlir::TypeRange{i32Ty},
        {mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrTy), slots, count,
         resultSlot});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, anyTy, resultSlot);
    return mlir::success();
  }
};

class ConvertTVMFFICastOp : public mlir::OpConversionPattern<tvm_ffi::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // All TVM FFI ABI semantic values are converted to the TVMFFIAny LLVM
    // struct by the type converter.  The dialect-level cast therefore becomes
    // an identity after conversion; the source conversion has already
    // materialized the correct type tag and payload.
    rewriter.replaceOp(op, adaptor.getValue());
    return mlir::success();
  }
};

class ConvertConstantOp
    : public mlir::OpConversionPattern<tvm_ffi::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ConstantOp op, tvm_ffi::ConstantOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::Type const resultType = op.getResult().getType();
    mlir::IntegerType const i64Ty = rewriter.getI64Type();
    mlir::Value payload;
    int32_t typeIndex;
    if (mlir::BoolAttr const attr =
            mlir::dyn_cast<mlir::BoolAttr>(op.getValue())) {
      typeIndex = tvm_ffi::BoolType::getTypeIndex();
      payload =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, attr.getValue());
    } else if (mlir::IntegerAttr const attr =
                   mlir::dyn_cast<mlir::IntegerAttr>(op.getValue())) {
      typeIndex = mlir::isa<tvm_ffi::DeviceType>(resultType)
                      ? tvm_ffi::DeviceType::getTypeIndex()
                      : tvm_ffi::IntType::getTypeIndex();
      payload =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, attr.getInt());
    } else if (mlir::ArrayAttr attr =
                   mlir::dyn_cast<mlir::ArrayAttr>(op.getValue());
               attr && mlir::isa<tvm_ffi::DTypeType>(resultType)) {
      if (attr.size() != 3 || !llvm::all_of(attr, [](mlir::Attribute value) {
            return mlir::isa<mlir::IntegerAttr>(value);
          })) {
        return op.emitError("dtype constant requires [code, bits, lanes]");
      }
      const int64_t code =
          mlir::cast<mlir::IntegerAttr>(attr[0]).getInt() & 0xff;
      const int64_t bits =
          mlir::cast<mlir::IntegerAttr>(attr[1]).getInt() & 0xff;
      const int64_t lanes =
          mlir::cast<mlir::IntegerAttr>(attr[2]).getInt() & 0xffff;
      const int64_t packed = code | (bits << 8) | (lanes << 16);
      typeIndex = tvm_ffi::DTypeType::getTypeIndex();
      payload = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, packed);
    } else if (mlir::StringAttr const attr =
                   mlir::dyn_cast<mlir::StringAttr>(op.getValue());
               attr && mlir::isa<tvm_ffi::DeviceType>(resultType)) {
      c10::Device const device = attr.getValue().str();
      DLDevice const dlDevice = at::torchDeviceToDLDevice(device);
      typeIndex = tvm_ffi::DeviceType::getTypeIndex();
      mlir::LLVM::LLVMStructType const dlDeviceTy =
          conversion::utils::getDLDeviceType(rewriter.getContext());
      mlir::Value deviceValue =
          mlir::LLVM::UndefOp::create(rewriter, loc, dlDeviceTy);
      deviceValue = mlir::LLVM::InsertValueOp::create(
          rewriter, loc, deviceValue,
          mlir::LLVM::ConstantOp::create(
              rewriter, loc, rewriter.getI32Type(),
              static_cast<int32_t>(dlDevice.device_type)),
          llvm::ArrayRef<int64_t>{0});
      deviceValue = mlir::LLVM::InsertValueOp::create(
          rewriter, loc, deviceValue,
          mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                         dlDevice.device_id),
          llvm::ArrayRef<int64_t>{1});
      mlir::LLVM::LLVMPointerType const ptrTy =
          mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
      mlir::Value const deviceSlot = mlir::LLVM::AllocaOp::create(
          rewriter, loc, ptrTy, dlDeviceTy,
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
      mlir::LLVM::StoreOp::create(rewriter, loc, deviceValue, deviceSlot);
      payload = mlir::LLVM::LoadOp::create(rewriter, loc, i64Ty, deviceSlot);
    } else if (mlir::StringAttr const attr =
                   mlir::dyn_cast<mlir::StringAttr>(op.getValue());
               attr && mlir::isa<tvm_ffi::RawStrType>(resultType)) {
      mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
      if (!module) {
        return op.emitError("failed to get parent ModuleOp");
      }
      mlir::Value const stringPtr = conversion::utils::getOrCreateGlobalString(
          rewriter, loc, module, "string", attr.getValue());
      typeIndex = tvm_ffi::RawStrType::getTypeIndex();
      payload = mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, stringPtr);
    } else if (mlir::FloatAttr const attr =
                   mlir::dyn_cast<mlir::FloatAttr>(op.getValue())) {
      typeIndex = tvm_ffi::FloatType::getTypeIndex();
      mlir::Value const value = mlir::LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getF64Type(),
          rewriter.getFloatAttr(rewriter.getF64Type(),
                                attr.getValueAsDouble()));
      payload = mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty, value);
    } else if (mlir::isa<mlir::UnitAttr>(op.getValue())) {
      typeIndex = tvm_ffi::NoneType::getTypeIndex();
      payload = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
    } else {
      return op.emitError("unsupported tvm_ffi.constant attribute");
    }
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFIABIType::getLLVMType(rewriter.getContext());
    mlir::IntegerType const i32Ty = rewriter.getI32Type();
    mlir::Value result = mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, result,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, typeIndex),
        llvm::ArrayRef<int64_t>{0});
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, result,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0),
        llvm::ArrayRef<int64_t>{1});
    result = mlir::LLVM::InsertValueOp::create(rewriter, loc, result, payload,
                                               llvm::ArrayRef<int64_t>{2});
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ConvertGetOp : public mlir::OpConversionPattern<tvm_ffi::GetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::GetOp op, tvm_ffi::GetOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const payload = mlir::LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), adaptor.getOperand(),
        llvm::ArrayRef<int64_t>{2});
    rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(op, rewriter.getI1Type(),
                                                     payload);
    return mlir::success();
  }
};

class ConvertEqOp : public mlir::OpConversionPattern<tvm_ffi::EqOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::EqOp op, tvm_ffi::EqOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::IntegerType const i32Ty = rewriter.getI32Type();
    mlir::IntegerType const i64Ty = rewriter.getI64Type();
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }

    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFIABIType::getLLVMType(rewriter.getContext());
    mlir::Value const one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
    mlir::Value const lhsStorage =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::Value const rhsStorage =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getLhs(), lhsStorage);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getRhs(), rhsStorage);
    mlir::Value const mapFreeVars = tvm_ffi::IntType::build(rewriter, loc, 0);
    mlir::Value const skipTensorContent =
        tvm_ffi::IntType::build(rewriter, loc, 0);
    mlir::FailureOr<mlir::Value> result =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "ffi.StructuralEqual",
            {lhsStorage, rhsStorage, mapFreeVars, skipTensorContent});
    if (mlir::failed(result)) {
      return op.emitError("failed to call ffi.StructuralEqual");
    }
    mlir::Value const resultPayloadI64 =
        tvm_ffi::TVMFFIABIType::load(rewriter, loc, result.value());
    mlir::Value const resultPayload =
        mlir::LLVM::TruncOp::create(rewriter, loc, i32Ty, resultPayloadI64);
    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
        op, mlir::arith::CmpIPredicate::ne, resultPayload,
        mlir::arith::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(0)));
    return mlir::success();
  }
};

class ConvertExceptionOp
    : public mlir::OpConversionPattern<tvm_ffi::ExceptionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ExceptionOp op, tvm_ffi::ExceptionOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Location const loc = op.getLoc();
    mlir::IntegerType const i32Ty = rewriter.getIntegerType(32);
    mlir::IntegerType const i64Ty = rewriter.getIntegerType(64);
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(context);
    mlir::LLVM::LLVMStructType const anyTy =
        tvm_ffi::TVMFFIABIType::getLLVMType(context);
    mlir::Value const kindPtr = conversion::utils::getOrCreateGlobalString(
        rewriter, loc, module, "ExceptionKind", op.getKind());
    mlir::Value exceptionArg =
        mlir::LLVM::UndefOp::create(rewriter, loc, anyTy);
    exceptionArg = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, exceptionArg,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 8),
        llvm::ArrayRef<int64_t>{0});
    exceptionArg = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, exceptionArg,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i32Ty, 0),
        llvm::ArrayRef<int64_t>{1});
    exceptionArg = mlir::LLVM::InsertValueOp::create(
        rewriter, loc, exceptionArg,
        mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, kindPtr),
        llvm::ArrayRef<int64_t>{2});
    mlir::Value const kindSlot = mlir::LLVM::AllocaOp::create(
        rewriter, loc, ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1));
    mlir::LLVM::StoreOp::create(rewriter, loc, exceptionArg, kindSlot);
    mlir::FailureOr<mlir::Value> resultSlot =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "trident.ffi.Exception",
            llvm::ArrayRef<mlir::Value>{kindSlot});
    if (mlir::failed(resultSlot)) {
      return op.emitError("failed to create TVM FFI exception");
    }
    mlir::Value const exception =
        mlir::LLVM::LoadOp::create(rewriter, loc, anyTy, resultSlot.value())
            .getResult();
    rewriter.replaceOp(op, exception);
    return mlir::success();
  }
};

class ConvertFunctionCallOp
    : public mlir::OpConversionPattern<tvm_ffi::FunctionCallOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::FunctionCallOp op,
                  tvm_ffi::FunctionCallOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1) {
      return op.emitError("tvm_ffi.FunctionCall currently requires one result");
    }
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    mlir::LLVM::LLVMStructType anyTy =
        tvm_ffi::TVMFFIABIType::getLLVMType(rewriter.getContext());
    mlir::LLVM::LLVMPointerType ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::IntegerType const i64Ty = rewriter.getI64Type();
    mlir::Value const count = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Ty,
        static_cast<int64_t>(adaptor.getArguments().size()));
    mlir::Value slots = mlir::LLVM::AllocaOp::create(rewriter, op.getLoc(),
                                                     ptrTy, anyTy, count);
    llvm::SmallVector<mlir::Value> const slotPtrs = llvm::map_to_vector(
        llvm::enumerate(adaptor.getArguments()),
        [&](auto indexedArgument) -> mlir::Value {
          auto [i, argument] = indexedArgument;
          mlir::Value slot = mlir::LLVM::GEPOp::create(
              rewriter, op.getLoc(), ptrTy, anyTy, slots,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
          mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), argument, slot);
          return slot;
        });
    mlir::Value const resultSlot = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, anyTy,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 1));
    mlir::Value const zero32 = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(), 0);
    mlir::Value const zero64 =
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 0);
    mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(), zero32,
        mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, anyTy,
                                  resultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
    mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(), zero32,
        mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, anyTy,
                                  resultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));
    mlir::LLVM::StoreOp::create(
        rewriter, op.getLoc(), zero64,
        mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, anyTy,
                                  resultSlot,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
    if (mlir::failed(conversion::utils::callTVMFFIFunction(
            rewriter, op.getLoc(), module, adaptor.getCallee(), slotPtrs,
            resultSlot))) {
      return op.emitError("failed to lower TVM FFI function call");
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, anyTy, resultSlot);
    return mlir::success();
  }
};

class ConvertFunctionGetGlobalOp
    : public mlir::OpConversionPattern<tvm_ffi::FunctionGetGlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::FunctionGetGlobalOp op,
                  tvm_ffi::FunctionGetGlobalOpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    mlir::FailureOr<mlir::Value> handle =
        conversion::utils::getTVMFFIGlobalFunction(rewriter, op.getLoc(),
                                                   module, op.getName());
    if (mlir::failed(handle)) {
      return op.emitError("failed to get TVM FFI global function");
    }
    rewriter.replaceOp(op, handle.value());
    return mlir::success();
  }
};

template <typename Op, auto GetCallee>
class ConvertRefOp : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::ModuleOp module = op->template getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return op.emitError("failed to get parent ModuleOp");
    }
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> callee = GetCallee(module);
    if (mlir::failed(callee)) {
      return op.emitError("failed to declare TVM FFI reference operation");
    }

    mlir::Value const payload = mlir::LLVM::ExtractValueOp::create(
        rewriter, loc, adaptor.getObject(), llvm::ArrayRef<int64_t>{2});
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Value const handle =
        mlir::LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, payload);

    // An !tvm_ffi.any value can contain a scalar, so inspect its ABI
    // TypeIndex before passing the payload to ObjectIncRef/ObjectDecRef.
    // Statically typed TVM FFI objects retain the direct call path.
    if (mlir::isa<tvm_ffi::AnyType, tvm_ffi::UnionType>(
            op.getObject().getType())) {
      mlir::Value const typeIndex = mlir::LLVM::ExtractValueOp::create(
          rewriter, loc, rewriter.getI32Type(), adaptor.getObject(),
          llvm::ArrayRef<int64_t>{0});
      mlir::Value const isObject = mlir::LLVM::ICmpOp::create(
          rewriter, loc, mlir::LLVM::ICmpPredicate::uge, typeIndex,
          mlir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                         kTVMFFIStaticObjectBegin));
      mlir::Block *currentBlock = rewriter.getInsertionBlock();
      mlir::Block *continuation =
          rewriter.splitBlock(currentBlock, op->getIterator());
      mlir::Block *callBlock = rewriter.createBlock(
          currentBlock->getParent(), continuation->getIterator());
      rewriter.setInsertionPointToEnd(currentBlock);
      mlir::LLVM::CondBrOp::create(rewriter, loc, isObject, callBlock,
                                   continuation);
      rewriter.setInsertionPointToEnd(callBlock);
      mlir::LLVM::CallOp::create(rewriter, loc, callee.value(),
                                 mlir::ValueRange{handle});
      mlir::LLVM::BrOp::create(rewriter, loc, continuation);
      rewriter.eraseOp(op);
    } else {
      mlir::LLVM::CallOp::create(rewriter, loc, callee.value(),
                                 mlir::ValueRange{handle});
      rewriter.eraseOp(op);
    }
    return mlir::success();
  }
};

/// Extract a DLTensor pointer from an SSA TVMFFIAny value.

static mlir::Value getDLTensorPtrFromAny(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value value) {
  mlir::MLIRContext *ctx = builder.getContext();
  mlir::IntegerType const i8Ty = mlir::IntegerType::get(ctx, 8);
  mlir::IntegerType const i64Ty = mlir::IntegerType::get(ctx, 64);
  mlir::LLVM::LLVMPointerType const ptrTy =
      mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Value const handleInt = mlir::LLVM::ExtractValueOp::create(
      builder, loc, i64Ty, value, llvm::ArrayRef<int64_t>{2});
  mlir::Value const handle =
      mlir::LLVM::IntToPtrOp::create(builder, loc, ptrTy, handleInt);
  return mlir::LLVM::GEPOp::create(
      builder, loc, ptrTy, i8Ty, handle,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{sizeof(TVMFFIObject)});
}

class ConvertTensorDeviceOp
    : public mlir::OpConversionPattern<tvm_ffi::TensorDeviceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::TensorDeviceOp op,
                  tvm_ffi::TensorDeviceOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType const tensorTy =
        conversion::utils::getDLTensorType(ctx);
    mlir::LLVM::LLVMStructType deviceTy =
        conversion::utils::getDLDeviceType(ctx);
    mlir::Value const tensor =
        getDLTensorPtrFromAny(rewriter, loc, adaptor.getTensor());
    mlir::Value devicePtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, tensorTy, tensor,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1});
    llvm::SmallVector<mlir::Value> const values = llvm::map_to_vector(
        llvm::seq<int64_t>(2), [&](int64_t index) -> mlir::Value {
          mlir::Value const fieldPtr = mlir::LLVM::GEPOp::create(
              rewriter, loc, ptrTy, deviceTy, devicePtr,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{0,
                                                 static_cast<int32_t>(index)});
          return mlir::LLVM::LoadOp::create(rewriter, loc,
                                            rewriter.getI32Type(), fieldPtr);
        });
    rewriter.replaceOp(op, values);
    return mlir::success();
  }
};

class ConvertTensorDimOp
    : public mlir::OpConversionPattern<tvm_ffi::TensorDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::TensorDimOp op, tvm_ffi::TensorDimOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType const tensorTy =
        conversion::utils::getDLTensorType(ctx);
    mlir::Value const tensor =
        getDLTensorPtrFromAny(rewriter, loc, adaptor.getTensor());
    mlir::Value const ptr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, tensorTy, tensor,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2});
    mlir::Value const value =
        mlir::LLVM::LoadOp::create(rewriter, loc, rewriter.getI32Type(), ptr);
    rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(op, rewriter.getI64Type(),
                                                    value);
    return mlir::success();
  }
};

class ConvertTensorDTypeOp
    : public mlir::OpConversionPattern<tvm_ffi::TensorDTypeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::TensorDTypeOp op,
                  tvm_ffi::TensorDTypeOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType const tensorTy =
        conversion::utils::getDLTensorType(ctx);
    mlir::LLVM::LLVMStructType const dtypeTy =
        conversion::utils::getDLDataType(ctx);
    mlir::Value const tensor =
        getDLTensorPtrFromAny(rewriter, loc, adaptor.getTensor());
    mlir::Value const dtypePtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, tensorTy, tensor,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 3});
    mlir::Value const code = mlir::LLVM::LoadOp::create(
        rewriter, loc, rewriter.getI8Type(),
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, dtypeTy, dtypePtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0}));
    mlir::Value const bits = mlir::LLVM::LoadOp::create(
        rewriter, loc, rewriter.getI8Type(),
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, dtypeTy, dtypePtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}));
    mlir::Value const lanes = mlir::LLVM::LoadOp::create(
        rewriter, loc, rewriter.getI16Type(),
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, dtypeTy, dtypePtr,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 2}));
    rewriter.replaceOp(op, mlir::ValueRange{code, bits, lanes});
    return mlir::success();
  }
};

template <typename SourceOp, int64_t Field>
class ConvertTensorIndexedMetadataOp
    : public mlir::OpConversionPattern<SourceOp> {
public:
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType const tensorTy =
        conversion::utils::getDLTensorType(ctx);
    mlir::Value const tensor =
        getDLTensorPtrFromAny(rewriter, loc, adaptor.getTensor());
    mlir::Value const valuesPtrPtr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, tensorTy, tensor,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, Field});
    mlir::Value const valuesPtr =
        mlir::LLVM::LoadOp::create(rewriter, loc, ptrTy, valuesPtrPtr);
    mlir::Value const elementPtr = mlir::LLVM::GEPOp::create(
        rewriter, loc, ptrTy, rewriter.getI64Type(), valuesPtr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{adaptor.getIndex()});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, rewriter.getI64Type(),
                                                    elementPtr);
    return mlir::success();
  }
};

class ConvertTensorStorageOffsetOp
    : public mlir::OpConversionPattern<tvm_ffi::TensorStorageOffsetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::TensorStorageOffsetOp op,
                  tvm_ffi::TensorStorageOffsetOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::LLVM::LLVMPointerType const ptrTy =
        mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::LLVM::LLVMStructType const tensorTy =
        conversion::utils::getDLTensorType(ctx);
    mlir::Value const tensor =
        getDLTensorPtrFromAny(rewriter, loc, adaptor.getTensor());
    mlir::Value const ptr =
        mlir::LLVM::GEPOp::create(rewriter, loc, ptrTy, tensorTy, tensor,
                                  llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 6});
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, rewriter.getI64Type(),
                                                    ptr);
    return mlir::success();
  }
};

/// Lowers TVM FFI operations and Any ABI values in ordinary func.func
/// operations. Control flow and function bodies are lowered by later passes.
class ConvertTVMFFIToLLVMPass
    : public impl::ConvertTVMFFIToLLVMBase<ConvertTVMFFIToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::RewritePatternSet patterns(&context);

    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);
    target.addLegalDialect<
        mlir::BuiltinDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
        mlir::arith::ArithDialect, mlir::gpu::GPUDialect, mlir::scf::SCFDialect,
        mlir::cf::ControlFlowDialect, mlir::torch::Torch::TorchDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) {
          return llvm::all_of(op.getOperandTypes(),
                              [&](mlir::Type type) {
                                return typeConverter.isLegal(type);
                              }) &&
                 llvm::all_of(op.getResultTypes(), [&](mlir::Type type) {
                   return typeConverter.isLegal(type);
                 });
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return mlir::isLegalForReturnOpTypeConversionPattern(op,
                                                               typeConverter);
        });

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
    populateTVMFFIToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

void populateTVMFFIToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  typeConverter.addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
    if (mlir::isa<tvm_ffi::FunctionType>(type)) {
      return mlir::LLVM::LLVMPointerType::get(type.getContext());
    } else if (type.hasTrait<mlir::TypeTrait::TVMFFIABI>()) {
      return tvm_ffi::TVMFFIABIType::getLLVMType(type.getContext());
    } else {
      return std::nullopt;
    }
  });
  patterns.add<ConvertArrayLengthOp, ConvertCallOp, ConvertTVMFFICastOp,
               ConvertConstantOp, ConvertEqOp, ConvertExceptionOp,
               ConvertFunctionCallOp, ConvertFunctionGetGlobalOp, ConvertGetOp,
               ConvertTensorDeviceOp, ConvertTensorDimOp, ConvertTensorDTypeOp,
               ConvertTensorLiteralOp,
               ConvertTensorIndexedMetadataOp<tvm_ffi::TensorSizeOp, 4>,
               ConvertTensorIndexedMetadataOp<tvm_ffi::TensorStrideOp, 5>,
               ConvertTensorStorageOffsetOp>(typeConverter,
                                             patterns.getContext());
  patterns.add<ConvertRefOp<tvm_ffi::ObjectDecRefOp,
                            &conversion::utils::getOrCreateTVMFFIObjectDecRef>,
               ConvertRefOp<tvm_ffi::ObjectIncRefOp,
                            &conversion::utils::getOrCreateTVMFFIObjectIncRef>>(
      typeConverter, patterns.getContext());
  target.addIllegalDialect<tvm_ffi::TVMFFIDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *, tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace trident::conversion
