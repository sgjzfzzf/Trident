//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TVMFFIToLLVM/TVMFFIToLLVM.h"
#include "ATen/dlpack.h"
#include "c10/core/Device.h"
#include "torch/headeronly/macros/Export.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/GlobalString.h"
#include "trident/core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
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
#include <tvm/ffi/c_api.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace at {
TORCH_API DLDevice torchDeviceToDLDevice(c10::Device device);
} // namespace at

namespace trident::tvm_ffi {

#define GEN_PASS_DEF_CONVERTTVMFFITOLLVM
#include "trident/core/Conversion/Passes.h.inc"

class ConvertArrayLengthOp : public mlir::OpConversionPattern<ArrayLengthOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayLengthOp op, OpAdaptor adaptor,
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
        conversion::utils::getTVMFFIAnyType(ctx);
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

class ConvertCallOp : public mlir::OpConversionPattern<CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    mlir::LLVM::LLVMStructType const anyTy =
        conversion::utils::getTVMFFIAnyType(ctx);
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

class ConvertCastOp : public mlir::OpConversionPattern<CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // All TVM FFI ABI semantic values are converted to the TVMFFIAny LLVM
    // struct by the type converter.  The dialect-level cast therefore becomes
    // an identity after conversion; the source conversion has already
    // materialized the correct type tag and payload.
    rewriter.replaceOp(op, adaptor.getValue());
    return mlir::success();
  }
};

class ConvertConstantOp : public mlir::OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location const loc = op.getLoc();
    mlir::Type const resultType = op.getResult().getType();
    mlir::IntegerType const i64Ty = rewriter.getI64Type();
    mlir::Value payload;
    int32_t typeIndex;
    if (mlir::BoolAttr const attr =
            mlir::dyn_cast<mlir::BoolAttr>(op.getValue())) {
      typeIndex = BoolType::getTypeIndex();
      payload =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, attr.getValue());
    } else if (mlir::IntegerAttr const attr =
                   mlir::dyn_cast<mlir::IntegerAttr>(op.getValue())) {
      typeIndex = mlir::isa<DeviceType>(resultType) ? DeviceType::getTypeIndex()
                                                    : IntType::getTypeIndex();
      payload =
          mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, attr.getInt());
    } else if (mlir::ArrayAttr attr =
                   mlir::dyn_cast<mlir::ArrayAttr>(op.getValue());
               attr && mlir::isa<DTypeType>(resultType)) {
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
      typeIndex = DTypeType::getTypeIndex();
      payload = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, packed);
    } else if (mlir::StringAttr const attr =
                   mlir::dyn_cast<mlir::StringAttr>(op.getValue());
               attr && mlir::isa<DeviceType>(resultType)) {
      c10::Device const device = attr.getValue().str();
      DLDevice const dlDevice = at::torchDeviceToDLDevice(device);
      typeIndex = DeviceType::getTypeIndex();
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
               attr && mlir::isa<RawStrType>(resultType)) {
      mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
      if (!module) {
        return op.emitError("failed to get parent ModuleOp");
      }
      mlir::Value const stringPtr = conversion::utils::getOrCreateGlobalString(
          rewriter, loc, module, "string", attr.getValue());
      typeIndex = RawStrType::getTypeIndex();
      payload = mlir::LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, stringPtr);
    } else if (mlir::FloatAttr const attr =
                   mlir::dyn_cast<mlir::FloatAttr>(op.getValue())) {
      typeIndex = FloatType::getTypeIndex();
      mlir::Value const value = mlir::LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getF64Type(),
          rewriter.getFloatAttr(rewriter.getF64Type(),
                                attr.getValueAsDouble()));
      payload = mlir::LLVM::BitcastOp::create(rewriter, loc, i64Ty, value);
    } else if (mlir::isa<mlir::UnitAttr>(op.getValue())) {
      typeIndex = NoneType::getTypeIndex();
      payload = mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0);
    } else {
      return op.emitError("unsupported tvm_ffi.constant attribute");
    }
    mlir::LLVM::LLVMStructType const anyTy =
        trident::conversion::utils::getTVMFFIAnyType(rewriter.getContext());
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

class ConvertEqOp : public mlir::OpConversionPattern<EqOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(EqOp op, OpAdaptor adaptor,
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
        trident::conversion::utils::getTVMFFIAnyType(rewriter.getContext());
    mlir::Value const one =
        mlir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 1);
    mlir::Value const lhsStorage =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::Value const rhsStorage =
        mlir::LLVM::AllocaOp::create(rewriter, loc, ptrTy, anyTy, one);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getLhs(), lhsStorage);
    mlir::LLVM::StoreOp::create(rewriter, loc, adaptor.getRhs(), rhsStorage);
    mlir::Value const mapFreeVars =
        conversion::utils::buildIntAnySlot(rewriter, loc, 0);
    mlir::Value const skipTensorContent =
        conversion::utils::buildIntAnySlot(rewriter, loc, 0);
    mlir::FailureOr<mlir::Value> result =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, loc, module, "ffi.StructuralEqual",
            {lhsStorage, rhsStorage, mapFreeVars, skipTensorContent});
    if (mlir::failed(result)) {
      return op.emitError("failed to call ffi.StructuralEqual");
    }
    mlir::Value const resultPayload =
        conversion::utils::loadIntFromAnySlot(rewriter, loc, result.value());
    rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
        op, mlir::arith::CmpIPredicate::ne, resultPayload,
        mlir::arith::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(0)));
    return mlir::success();
  }
};

class ConvertExceptionOp : public mlir::OpConversionPattern<ExceptionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ExceptionOp op, OpAdaptor,
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
        conversion::utils::getTVMFFIAnyType(context);
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

class ConvertFunctionCallOp : public mlir::OpConversionPattern<FunctionCallOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(FunctionCallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1) {
      return op.emitError("tvm_ffi.FunctionCall currently requires one result");
    }
    mlir::ModuleOp const module = op->getParentOfType<mlir::ModuleOp>();
    mlir::LLVM::LLVMStructType anyTy =
        conversion::utils::getTVMFFIAnyType(rewriter.getContext());
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
    : public mlir::OpConversionPattern<FunctionGetGlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(FunctionGetGlobalOp op, OpAdaptor,
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
    if (mlir::isa<AnyType, UnionType>(op.getObject().getType())) {
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

class ConvertTensorDeviceOp : public mlir::OpConversionPattern<TensorDeviceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorDeviceOp op, OpAdaptor adaptor,
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

class ConvertTensorDimOp : public mlir::OpConversionPattern<TensorDimOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorDimOp op, OpAdaptor adaptor,
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

class ConvertTensorDTypeOp : public mlir::OpConversionPattern<TensorDTypeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorDTypeOp op, OpAdaptor adaptor,
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
    : public mlir::OpConversionPattern<TensorStorageOffsetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorStorageOffsetOp op, OpAdaptor adaptor,
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
    if (mlir::isa<FunctionType>(type)) {
      return mlir::LLVM::LLVMPointerType::get(type.getContext());
    } else if (type.hasTrait<mlir::TypeTrait::TVMFFIABI>()) {
      return trident::conversion::utils::getTVMFFIAnyType(type.getContext());
    } else {
      return std::nullopt;
    }
  });
  patterns.add<ConvertArrayLengthOp, ConvertCallOp, ConvertCastOp,
               ConvertConstantOp, ConvertEqOp, ConvertExceptionOp,
               ConvertFunctionCallOp, ConvertFunctionGetGlobalOp,
               ConvertTensorDeviceOp, ConvertTensorDimOp, ConvertTensorDTypeOp,
               ConvertTensorIndexedMetadataOp<TensorSizeOp, 4>,
               ConvertTensorIndexedMetadataOp<TensorStrideOp, 5>,
               ConvertTensorStorageOffsetOp>(typeConverter,
                                             patterns.getContext());
  patterns.add<ConvertRefOp<ObjectDecRefOp,
                            &conversion::utils::getOrCreateTVMFFIObjectDecRef>,
               ConvertRefOp<ObjectIncRefOp,
                            &conversion::utils::getOrCreateTVMFFIObjectIncRef>>(
      typeConverter, patterns.getContext());
  target.addIllegalDialect<TVMFFIDialect>();
}

void registerConvertTVMFFIToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *, tvm_ffi::TVMFFIDialect *dialect) {
        dialect->addInterfaces<TVMFFIToLLVMDialectInterface>();
      });
}

} // namespace trident::tvm_ffi
