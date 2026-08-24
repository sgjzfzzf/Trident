//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/TridentCAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <type_traits>

namespace trident::torch {
#define GEN_PASS_DEF_CONVERTTORCHTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"

namespace {

using OwnedValues =
    llvm::DenseMap<mlir::Region *, llvm::SmallSetVector<mlir::Value, 4>>;

static void recordOwnedObjectResults(mlir::Operation *operation,
                                     mlir::ValueRange results,
                                     OwnedValues &ownedValues) {
  for (mlir::Value result : results) {
    if (result.getType().hasTrait<mlir::TypeTrait::Object>()) {
      ownedValues[operation->getParentRegion()].insert(result);
    }
  }
}

/// TypeConverter used by this bridge.  Keeping the Torch-to-semantic mapping
/// in a TypeConverter makes it reusable by conversion patterns and gives us
/// the standard MLIR materialization path for the temporary boundary casts.
class TorchFFITypeConverter : public mlir::TypeConverter {
public:
  TorchFFITypeConverter() {
    addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
      mlir::MLIRContext *ctx = type.getContext();
      if (mlir::isa<tvm_ffi::AnyType, tvm_ffi::ExceptionType, tvm_ffi::BoolType,
                    tvm_ffi::IntType, tvm_ffi::FloatType, tvm_ffi::NoneType,
                    tvm_ffi::DeviceType, tvm_ffi::ArrayType,
                    tvm_ffi::TensorType>(type)) {
        return type;
      } else if (mlir::isa<mlir::torch::Torch::BoolType>(type)) {
        return tvm_ffi::BoolType::get(ctx);
      } else if (mlir::isa<mlir::torch::Torch::IntType>(type)) {
        return tvm_ffi::IntType::get(ctx);
      } else if (mlir::isa<mlir::torch::Torch::FloatType>(type)) {
        return tvm_ffi::FloatType::get(ctx);
      } else if (mlir::isa<mlir::torch::Torch::NoneType>(type)) {
        return tvm_ffi::NoneType::get(ctx);
      } else if (mlir::isa<mlir::torch::Torch::DeviceType>(type)) {
        return tvm_ffi::DeviceType::get(ctx);
      } else if (mlir::isa<mlir::torch::Torch::NonValueTensorType,
                           mlir::torch::Torch::ValueTensorType>(type)) {
        return tvm_ffi::TensorType::get(ctx);
      } else if (mlir::isa<mlir::torch::Torch::ListType,
                           mlir::torch::Torch::TupleType>(type)) {
        return tvm_ffi::ArrayType::get(ctx);
      } else if (mlir::isa<mlir::torch::Torch::AnyType>(type)) {
        return tvm_ffi::ArrayType::get(ctx);
      } else {
        return std::nullopt;
      }
    });
    addConversion([](mlir::IntegerType type) -> mlir::Type { return type; });
    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      return mlir::UnrealizedConversionCastOp::create(
                 builder, loc, mlir::TypeRange(type), inputs)
          .getResult(0);
    });
    addSourceMaterialization([](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      return mlir::UnrealizedConversionCastOp::create(
                 builder, loc, mlir::TypeRange(type), inputs)
          .getResult(0);
    });
  }
};

static void populateTerminatorRefCounts(
    mlir::Operation *terminator,
    const llvm::SmallSetVector<mlir::Value, 4> &ownedValues,
    const TorchFFITypeConverter &typeConverter) {
  mlir::OpBuilder builder(terminator);
  mlir::Location loc = terminator->getLoc();
  llvm::SmallSetVector<mlir::Value, 4> valuesToRelease(ownedValues);

  for (mlir::Value operand : terminator->getOperands()) {
    mlir::Value value = operand;
    mlir::UnrealizedConversionCastOp cast;
    while ((cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) &&
           cast->getNumOperands() == 1) {
      value = cast->getOperand(0);
    }
    mlir::Type convertedType = typeConverter.convertType(value.getType());
    if (convertedType && convertedType.hasTrait<mlir::TypeTrait::Object>()) {
      // A normal func.call returns an owned semantic value. Returning it from
      // another normal func.func transfers that ownership to the caller; an
      // additional IncRef here would leak one reference at every ABI wrapper
      // boundary. TVMFFI calls still require the existing retain/release pair
      // below because their result ownership is handled by the FFI bridge.
      if (value.getDefiningOp<mlir::func::CallOp>()) {
        continue;
      }
      tvm_ffi::ObjectIncRefOp::create(builder, loc, value);
      if (value.getDefiningOp<tvm_ffi::CallOp>()) {
        valuesToRelease.insert(value);
      }
    }
  }

  for (mlir::Value value : valuesToRelease) {
    tvm_ffi::ObjectDecRefOp::create(builder, loc, value);
  }
}

template <typename TerminatorOp>
class ConvertTerminatorOp final : public mlir::OpRewritePattern<TerminatorOp> {
public:
  ConvertTerminatorOp(
      mlir::MLIRContext *context,
      const llvm::DenseMap<mlir::Region *, llvm::SmallSetVector<mlir::Value, 4>>
          &ownedValues,
      const TorchFFITypeConverter &typeConverter)
      : mlir::OpRewritePattern<TerminatorOp>(context), ownedValues(ownedValues),
        typeConverter(typeConverter) {}

  mlir::LogicalResult matchAndRewrite(TerminatorOp op,
                                      mlir::PatternRewriter &) const override {
    mlir::Region *region = op->getParentRegion();
    if (!region || !region->hasOneBlock()) {
      return mlir::failure();
    }
    auto it = ownedValues.find(region);
    if (it != ownedValues.end()) {
      populateTerminatorRefCounts(op, it->second, typeConverter);
    } else {
      populateTerminatorRefCounts(op, {}, typeConverter);
    }
    return mlir::success();
  }

private:
  const llvm::DenseMap<mlir::Region *, llvm::SmallSetVector<mlir::Value, 4>>
      &ownedValues;
  const TorchFFITypeConverter &typeConverter;
};

template <typename ConstructOp>
class ConvertTorchArrayConstruct final
    : public mlir::OpConversionPattern<ConstructOp> {
public:
  explicit ConvertTorchArrayConstruct(
      const TorchFFITypeConverter &typeConverter, OwnedValues &ownedValues,
      mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ConstructOp>(typeConverter, ctx, 1),
        ownedValues(ownedValues) {}

  mlir::LogicalResult
  matchAndRewrite(ConstructOp op, typename ConstructOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    tvm_ffi::ArrayCreateOp array =
        rewriter.replaceOpWithNewOp<tvm_ffi::ArrayCreateOp>(
            op, tvm_ffi::ArrayType::get(this->getContext()),
            adaptor.getElements());
    recordOwnedObjectResults(array, array->getResults(), ownedValues);
    return mlir::success();
  }

private:
  llvm::DenseMap<mlir::Region *, llvm::SmallSetVector<mlir::Value, 4>>
      &ownedValues;
};

class ConvertTorchArrayUnpack final
    : public mlir::OpConversionPattern<mlir::torch::Torch::PrimListUnpackOp> {
public:
  explicit ConvertTorchArrayUnpack(const TorchFFITypeConverter &typeConverter,
                                   OwnedValues &, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::torch::Torch::PrimListUnpackOp>(
            typeConverter, ctx, 1),
        typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::PrimListUnpackOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value array = adaptor.getOperand();
    llvm::SmallVector<mlir::Value> replacements;
    replacements.reserve(op.getNumResults());
    for (auto [index, result] : llvm::enumerate(op.getResults())) {
      mlir::Type resultType = typeConverter.convertType(result.getType());
      if (!resultType) {
        return op.emitError("cannot convert ListUnpack result type");
      }
      tvm_ffi::ConstantOp idx = tvm_ffi::ConstantOp::create(
          rewriter, op.getLoc(), tvm_ffi::IntType::get(getContext()),
          rewriter.getI64IntegerAttr(index));
      tvm_ffi::ArrayGetItemOp item = tvm_ffi::ArrayGetItemOp::create(
          rewriter, op.getLoc(), resultType, array, idx.getResult(),
          mlir::TypeAttr::get(resultType));
      replacements.push_back(item.getResult());
    }
    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
};

/// Keep an already-semantic operation while converting its operands.
template <typename Op>
class ConvertGenericOp final : public mlir::OpConversionPattern<Op> {
public:
  ConvertGenericOp(const TorchFFITypeConverter &typeConverter,
                   mlir::MLIRContext *context)
      : mlir::OpConversionPattern<Op>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
};

template <typename ConstantOp>
class ConvertTorchConstant final
    : public mlir::OpConversionPattern<ConstantOp> {
public:
  explicit ConvertTorchConstant(const TorchFFITypeConverter &typeConverter,
                                OwnedValues &ownedValues,
                                mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ConstantOp>(typeConverter, ctx, 1),
        typeConverter(typeConverter), ownedValues(ownedValues) {}

  mlir::LogicalResult
  matchAndRewrite(ConstantOp op, typename ConstantOp::Adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type targetType = typeConverter.convertType(op.getResult().getType());
    if (!targetType) {
      return mlir::failure();
    }
    mlir::Attribute value = op->getAttr("value");
    if constexpr (std::is_same_v<ConstantOp,
                                 mlir::torch::Torch::ConstantNoneOp>)
      value = rewriter.getUnitAttr();
    if (!value) {
      return op->emitError("constant is missing value attribute");
    }
    tvm_ffi::ConstantOp constant =
        rewriter.replaceOpWithNewOp<tvm_ffi::ConstantOp>(op, targetType, value);
    recordOwnedObjectResults(constant, constant->getResults(), ownedValues);
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
  llvm::DenseMap<mlir::Region *, llvm::SmallSetVector<mlir::Value, 4>>
      &ownedValues;
};

class ConvertAtenCall final
    : public mlir::OpConversionPattern<mlir::torch::Torch::OperatorOp> {
public:
  explicit ConvertAtenCall(const TorchFFITypeConverter &typeConverter,
                           OwnedValues &ownedValues, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::torch::Torch::OperatorOp>(typeConverter,
                                                                  ctx, 1),
        typeConverter(typeConverter), ownedValues(ownedValues) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::OperatorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::StringRef name = op.getName();
    if (!name.starts_with("torch.aten.")) {
      return mlir::failure();
    }
    std::string callee =
        ("trident." + name.drop_front(sizeof("torch.") - 1)).str();
    mlir::ValueRange operands = adaptor.getOperands();
    llvm::SmallVector<mlir::Value> replacements;
    if (op->getNumResults() <= 1) {
      llvm::SmallVector<mlir::Type> resultTypes;
      if (op->getNumResults()) {
        mlir::Type resultType =
            typeConverter.convertType(op->getResult(0).getType());
        if (!resultType) {
          return mlir::failure();
        }
        resultTypes.push_back(resultType);
      }
      tvm_ffi::FunctionGetGlobalOp getGlobal =
          tvm_ffi::FunctionGetGlobalOp::create(
              rewriter, op->getLoc(), tvm_ffi::FunctionType::get(getContext()),
              callee);
      tvm_ffi::FunctionCallOp call = tvm_ffi::FunctionCallOp::create(
          rewriter, op->getLoc(), resultTypes, getGlobal.getResult(), operands);
      recordOwnedObjectResults(op.getOperation(), call->getResults(),
                               ownedValues);
      if (op->getNumResults()) {
        replacements.push_back(call.getResult(0));
      }
    } else {
      mlir::Type arrayType = tvm_ffi::ArrayType::get(getContext());
      tvm_ffi::FunctionGetGlobalOp getGlobal =
          tvm_ffi::FunctionGetGlobalOp::create(
              rewriter, op->getLoc(), tvm_ffi::FunctionType::get(getContext()),
              callee);
      tvm_ffi::FunctionCallOp call = tvm_ffi::FunctionCallOp::create(
          rewriter, op->getLoc(), mlir::TypeRange{arrayType},
          getGlobal.getResult(), operands);
      recordOwnedObjectResults(op.getOperation(), call->getResults(),
                               ownedValues);
      for (auto [index, result] : llvm::enumerate(op->getResults())) {
        tvm_ffi::ConstantOp idx = tvm_ffi::ConstantOp::create(
            rewriter, op->getLoc(), tvm_ffi::IntType::get(getContext()),
            rewriter.getI64IntegerAttr(index));
        mlir::Type base = typeConverter.convertType(result.getType());
        if (!base) {
          return mlir::failure();
        }
        mlir::Type semantic = base;

        tvm_ffi::ArrayGetItemOp item = tvm_ffi::ArrayGetItemOp::create(
            rewriter, op->getLoc(), base, call.getResult(0), idx.getResult(),
            mlir::TypeAttr::get(semantic));
        replacements.push_back(item.getResult());
      }
    }
    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
  llvm::DenseMap<mlir::Region *, llvm::SmallSetVector<mlir::Value, 4>>
      &ownedValues;
};

class ConvertTorchValueTensorLiteralOp final
    : public mlir::OpConversionPattern<
          mlir::torch::Torch::ValueTensorLiteralOp> {
public:
  ConvertTorchValueTensorLiteralOp(const TorchFFITypeConverter &typeConverter,
                                   mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::torch::Torch::ValueTensorLiteralOp>(
            typeConverter, ctx, 1) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::ValueTensorLiteralOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::DenseElementsAttr dense =
        mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    auto tensorType =
        mlir::dyn_cast<mlir::torch::Torch::BaseTensorType>(op.getType());
    if (!dense || !tensorType || !tensorType.hasSizes()) {
      return op.emitError("literal requires a dense tensor and static shape");
    }
    llvm::ArrayRef<int64_t> shape = tensorType.getSizes();
    mlir::RankedTensorType denseType =
        mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
    if (!denseType || denseType.getShape() != shape) {
      return op.emitError("literal shape does not match result type");
    }

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::ModuleOp module = op->template getParentOfType<mlir::ModuleOp>();
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::IntegerType i8Ty = mlir::IntegerType::get(ctx, 8);
    mlir::IntegerType i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMStructType anyTy = conversion::utils::getTVMFFIAnyType(ctx);

    int64_t elementCount = dense.getNumElements();
    int64_t elementBits = 0;
    int32_t dtypeCode = 0;
    if (mlir::FloatType floatType =
            mlir::dyn_cast<mlir::FloatType>(denseType.getElementType())) {
      elementBits = floatType.getWidth();
      dtypeCode = kDLFloat;
    } else if (mlir::IntegerType integerType =
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
    mlir::FailureOr<mlir::LLVM::LLVMFuncOp> dtypeFn =
        conversion::utils::getOrCreateTVMFFIToTorchType(module);
    if (mlir::failed(dtypeFn)) {
      return op.emitError("failed to declare tensor dtype conversion helper");
    }

    mlir::Value dtype = mlir::LLVM::CallOp::create(
                            rewriter, op.getLoc(), *dtypeFn,
                            {mlir::LLVM::ConstantOp::create(
                                 rewriter, op.getLoc(), i8Ty, dtypeCode),
                             mlir::LLVM::ConstantOp::create(
                                 rewriter, op.getLoc(), i8Ty, elementBits)})
                            .getResult();
    mlir::Value rank = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Ty, static_cast<int64_t>(shape.size()));
    mlir::Value sizes = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                       shape.size()));
    mlir::Value strides = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                       shape.size()));
    int64_t stride = 1;
    for (int64_t index = shape.size() - 1; index >= 0; --index) {
      mlir::Value sizeSlot =
          mlir::LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, i64Ty, sizes,
                                    llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
      mlir::Value strideSlot = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), ptrTy, i64Ty, strides,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
      mlir::LLVM::StoreOp::create(
          rewriter, op.getLoc(),
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                         shape[index]),
          sizeSlot);
      mlir::LLVM::StoreOp::create(
          rewriter, op.getLoc(),
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, stride),
          strideSlot);
      stride *= shape[index];
    }

    mlir::Type storageType = denseType.getElementType();
    if (mlir::IntegerType integerType =
            mlir::dyn_cast<mlir::IntegerType>(storageType);
        integerType && integerType.isSignlessInteger(1)) {
      storageType = i8Ty;
    }
    mlir::Value data = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, storageType,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                       elementCount));
    auto storeElement = [&](int64_t index, mlir::Value value) {
      mlir::Value slot = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), ptrTy, storageType, data,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{index});
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), value, slot);
    };
    if (mlir::isa<mlir::FloatType>(storageType)) {
      auto values = dense.getValues<llvm::APFloat>();
      for (auto [index, value] : llvm::enumerate(values)) {
        storeElement(index, mlir::LLVM::ConstantOp::create(
                                rewriter, op.getLoc(), storageType,
                                mlir::FloatAttr::get(storageType, value)));
      }
    } else {
      auto values = dense.getValues<llvm::APInt>();
      for (auto [index, value] : llvm::enumerate(values)) {
        if (mlir::cast<mlir::IntegerType>(denseType.getElementType())
                .isSignlessInteger(1)) {
          value = llvm::APInt(8, value.getBoolValue());
        }
        storeElement(index, mlir::LLVM::ConstantOp::create(
                                rewriter, op.getLoc(), storageType,
                                mlir::IntegerAttr::get(storageType, value)));
      }
    }

    mlir::Value tensorHandle;
    mlir::Value zeroI32 =
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, 0);
    mlir::Value oneI64 =
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 1);

    if (dense.isSplat()) {
      double fillValue;
      if (mlir::isa<mlir::FloatType>(storageType)) {
        fillValue = dense.getSplatValue<llvm::APFloat>().convertToDouble();
      } else {
        llvm::APInt value = dense.getSplatValue<llvm::APInt>();
        auto integerType = mlir::cast<mlir::IntegerType>(storageType);
        fillValue =
            integerType.isSignlessInteger(1)
                ? value.getBoolValue()
                : (integerType.isUnsignedInteger() ? value.getZExtValue()
                                                   : value.getSExtValue());
      }
      mlir::LLVM::LLVMFuncOp fullFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchAtenFull(module));
      mlir::LLVM::LLVMFuncOp getDeviceIndexFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(module));
      mlir::Value cudaDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, kDLCUDA);
      mlir::LLVM::LLVMFuncOp toDeviceFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateTVMFFIDeviceToTorchDeviceType(module));
      mlir::Value cudaDeviceType =
          mlir::LLVM::CallOp::create(rewriter, op.getLoc(), toDeviceFn,
                                     cudaDLDeviceType)
              .getResult();
      mlir::Value dtypeSlot = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, i32Ty, oneI64);
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), dtype, dtypeSlot);
      mlir::Value deviceSlot = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, i32Ty, oneI64);
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), cudaDeviceType,
                                  deviceSlot);
      mlir::Value deviceIndexSlot = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, i32Ty, oneI64);
      mlir::LLVM::CallOp::create(rewriter, op.getLoc(), getDeviceIndexFn,
                                 deviceIndexSlot);
      mlir::Value deviceIndex = mlir::LLVM::LoadOp::create(
          rewriter, op.getLoc(), i32Ty, deviceIndexSlot);
      mlir::Value output = mlir::LLVM::AllocaOp::create(rewriter, op.getLoc(),
                                                        ptrTy, ptrTy, oneI64);
      mlir::Value fill = mlir::LLVM::ConstantOp::create(
          rewriter, op.getLoc(), mlir::Float64Type::get(ctx),
          mlir::FloatAttr::get(mlir::Float64Type::get(ctx), fillValue));
      mlir::LLVM::CallOp::create(
          rewriter, op.getLoc(), fullFn,
          {sizes, rank, fill, dtypeSlot,
           mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrTy), deviceSlot,
           deviceIndex,
           mlir::LLVM::ZeroOp::create(rewriter, op.getLoc(), ptrTy), output});
      tensorHandle =
          mlir::LLVM::LoadOp::create(rewriter, op.getLoc(), ptrTy, output);
    } else {
      mlir::LLVM::LLVMFuncOp createFromBlobFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchCreateTensorFromBlob(module));
      mlir::LLVM::LLVMFuncOp emptyStridedFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchEmptyStrided(module));
      mlir::LLVM::LLVMFuncOp copyFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchCopy_(module));
      mlir::LLVM::LLVMFuncOp deleteTensorFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchDeleteTensorObject(module));
      mlir::LLVM::LLVMFuncOp getDeviceIndexFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(module));
      mlir::LLVM::LLVMFuncOp toDeviceFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateTVMFFIDeviceToTorchDeviceType(module));
      mlir::Value cpuDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, kDLCPU);
      mlir::Value cudaDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, kDLCUDA);
      mlir::Value cpuDeviceType =
          mlir::LLVM::CallOp::create(rewriter, op.getLoc(), toDeviceFn,
                                     cpuDLDeviceType)
              .getResult();
      mlir::Value cudaDeviceType =
          mlir::LLVM::CallOp::create(rewriter, op.getLoc(), toDeviceFn,
                                     cudaDLDeviceType)
              .getResult();
      mlir::Value deviceIndexSlot = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, i32Ty, oneI64);
      mlir::LLVM::CallOp::create(rewriter, op.getLoc(), getDeviceIndexFn,
                                 deviceIndexSlot);
      mlir::Value deviceIndex = mlir::LLVM::LoadOp::create(
          rewriter, op.getLoc(), i32Ty, deviceIndexSlot);
      mlir::Value cpuOutput = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, ptrTy, oneI64);
      mlir::LLVM::CallOp::create(
          rewriter, op.getLoc(), createFromBlobFn,
          {data, rank, sizes, strides,
           mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 0),
           dtype, cpuDeviceType, zeroI32, cpuOutput});
      mlir::Value cpuTensor =
          mlir::LLVM::LoadOp::create(rewriter, op.getLoc(), ptrTy, cpuOutput);
      mlir::Value cudaOutput = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, ptrTy, oneI64);
      mlir::LLVM::CallOp::create(rewriter, op.getLoc(), emptyStridedFn,
                                 {rank, sizes, strides, dtype, cudaDeviceType,
                                  deviceIndex, cudaOutput});
      tensorHandle =
          mlir::LLVM::LoadOp::create(rewriter, op.getLoc(), ptrTy, cudaOutput);
      mlir::LLVM::CallOp::create(rewriter, op.getLoc(), copyFn,
                                 {tensorHandle, cpuTensor, zeroI32});
      mlir::LLVM::CallOp::create(rewriter, op.getLoc(), deleteTensorFn,
                                 cpuTensor);
    }

    mlir::LLVM::LLVMFuncOp packFn = TRIDENT_CHECK_FAILURE(
        conversion::utils::getOrCreateTensorToTVMFFIObject(module));
    mlir::Value output = mlir::LLVM::AllocaOp::create(rewriter, op.getLoc(),
                                                      ptrTy, ptrTy, oneI64);
    mlir::LLVM::CallOp::create(rewriter, op.getLoc(), packFn,
                               {tensorHandle, output});
    mlir::Value handle =
        mlir::LLVM::LoadOp::create(rewriter, op.getLoc(), ptrTy, output);
    mlir::Value payload =
        mlir::LLVM::PtrToIntOp::create(rewriter, op.getLoc(), i64Ty, handle);
    mlir::Value result =
        mlir::LLVM::UndefOp::create(rewriter, op.getLoc(), anyTy);
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), result,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty,
                                       tvm_ffi::TensorType::getTypeIndex()),
        llvm::ArrayRef<int64_t>{0});
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), result,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, 0),
        llvm::ArrayRef<int64_t>{1});
    result = mlir::LLVM::InsertValueOp::create(
        rewriter, op.getLoc(), result, payload, llvm::ArrayRef<int64_t>{2});
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ConvertTVMFFIReturn final
    : public mlir::OpRewritePattern<tvm_ffi::ReturnOp> {
public:
  explicit ConvertTVMFFIReturn(const TorchFFITypeConverter &typeConverter,
                               mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<tvm_ffi::ReturnOp>(ctx, 1),
        typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ReturnOp ret,
                  mlir::PatternRewriter &rewriter) const override {
    tvm_ffi::FuncOp func = ret->getParentOfType<tvm_ffi::FuncOp>();
    if (!func) {
      return mlir::failure();
    }
    mlir::FunctionType type = func.getFunctionType();
    if (type.getNumResults() == 1 &&
        mlir::isa<tvm_ffi::AnyType>(type.getResult(0))) {
      return mlir::failure();
    }
    bool changed = false;
    llvm::SmallVector<mlir::Value> operands(ret.getOperands());
    for (auto [index, resultType] : llvm::enumerate(type.getResults())) {
      mlir::Value value = operands[index];
      if (value.getType() == resultType) {
        continue;
      }
      mlir::UnrealizedConversionCastOp cast;
      while (typeConverter.convertType(value.getType()) == resultType &&
             (cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) &&
             cast->getNumOperands() == 1) {
        value = cast->getOperand(0);
      }
      if (value.getType() != resultType) {
        value = typeConverter.materializeTargetConversion(
            rewriter, ret.getLoc(), resultType, value);
        if (!value) {
          return mlir::failure();
        }
      }
      operands[index] = value;
      changed = true;
    }
    if (!changed) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<tvm_ffi::ReturnOp>(ret, operands);
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
};

class ConvertTorchToTVMFFIPass final
    : public impl::ConvertTorchToTVMFFIBase<ConvertTorchToTVMFFIPass> {
  void runOnOperation() final {
    TorchFFITypeConverter typeConverter;
    OwnedValues ownedValues;

    mlir::RewritePatternSet conversionPatterns(&getContext());
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        conversionPatterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(conversionPatterns,
                                              typeConverter);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<tvm_ffi::FuncOp>(
        conversionPatterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(conversionPatterns,
                                                typeConverter);
    conversionPatterns.add<
        ConvertTorchArrayConstruct<mlir::torch::Torch::PrimListConstructOp>,
        ConvertTorchArrayConstruct<mlir::torch::Torch::PrimTupleConstructOp>,
        ConvertTorchArrayUnpack,
        ConvertTorchConstant<mlir::torch::Torch::ConstantNoneOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantBoolOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantIntOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantFloatOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantDeviceOp>,
        ConvertAtenCall>(typeConverter, ownedValues, &getContext());
    conversionPatterns.add<ConvertGenericOp<tvm_ffi::EqOp>,
                           ConvertGenericOp<tvm_ffi::TensorDimOp>,
                           ConvertGenericOp<tvm_ffi::TensorSizeOp>,
                           ConvertGenericOp<tvm_ffi::TensorStrideOp>,
                           ConvertGenericOp<tvm_ffi::TensorStorageOffsetOp>,
                           ConvertGenericOp<tvm_ffi::TensorDTypeOp>,
                           ConvertGenericOp<tvm_ffi::TensorDeviceOp>,
                           ConvertGenericOp<tvm_ffi::ArrayLengthOp>,
                           ConvertTorchValueTensorLiteralOp>(typeConverter,
                                                             &getContext());

    mlir::ConversionTarget conversionTarget(getContext());
    conversionTarget.addLegalDialect<
        mlir::BuiltinDialect, mlir::arith::ArithDialect,
        mlir::LLVM::LLVMDialect, trident::arithext::ArithExtDialect>();
    conversionTarget.addLegalOp<
        tvm_ffi::ArrayCreateOp, tvm_ffi::ArrayGetItemOp, tvm_ffi::CallOp,
        tvm_ffi::FunctionGetGlobalOp, tvm_ffi::FunctionCallOp,
        tvm_ffi::ConstantOp, tvm_ffi::CastOp, tvm_ffi::ObjectDecRefOp,
        tvm_ffi::ObjectIncRefOp, tvm_ffi::ExceptionOp>();
    conversionTarget.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp func) {
          return typeConverter.isSignatureLegal(func.getFunctionType());
        });
    conversionTarget.addDynamicallyLegalOp<tvm_ffi::FuncOp>(
        [&](tvm_ffi::FuncOp func) {
          return typeConverter.isSignatureLegal(func.getFunctionType());
        });
    conversionTarget.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp ret) {
          return mlir::isLegalForReturnOpTypeConversionPattern(ret,
                                                               typeConverter);
        });
    conversionTarget.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp call) {
          return llvm::all_of(call.getOperandTypes(),
                              [&](mlir::Type type) {
                                return typeConverter.isLegal(type);
                              }) &&
                 llvm::all_of(call.getResultTypes(), [&](mlir::Type type) {
                   return typeConverter.isLegal(type);
                 });
        });
    conversionTarget.addDynamicallyLegalOp<
        tvm_ffi::EqOp, tvm_ffi::TensorDimOp, tvm_ffi::TensorSizeOp,
        tvm_ffi::TensorStrideOp, tvm_ffi::TensorStorageOffsetOp,
        tvm_ffi::TensorDTypeOp, tvm_ffi::TensorDeviceOp,
        tvm_ffi::ArrayLengthOp>([&](mlir::Operation *op) {
      return llvm::all_of(op->getOperandTypes(), [&](mlir::Type type) {
        mlir::Type convertedType = typeConverter.convertType(type);
        return !convertedType || convertedType == type;
      });
    });
    conversionTarget
        .addLegalOp<mlir::ModuleOp, mlir::torch::Torch::PrimIfOp,
                    mlir::torch::Torch::PrimIfYieldOp, tvm_ffi::ReturnOp>();
    conversionTarget.addDynamicallyLegalOp<mlir::torch::Torch::OperatorOp>(
        [](mlir::torch::Torch::OperatorOp op) {
          return !op.getName().starts_with("torch.aten.");
        });
    conversionTarget.addIllegalOp<mlir::torch::Torch::ValueTensorLiteralOp>();
    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), conversionTarget, std::move(conversionPatterns)))) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet terminatorPatterns(&getContext());
    terminatorPatterns.add<ConvertTVMFFIReturn>(typeConverter, &getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(terminatorPatterns)))) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet refCountPatterns(&getContext());
    refCountPatterns
        .add<ConvertTerminatorOp<mlir::func::ReturnOp>,
             ConvertTerminatorOp<tvm_ffi::ReturnOp>,
             ConvertTerminatorOp<mlir::torch::Torch::PrimIfYieldOp>>(
            &getContext(), ownedValues, typeConverter);
    mlir::FrozenRewritePatternSet frozenRefCountPatterns(
        std::move(refCountPatterns));
    mlir::walkAndApplyPatterns(getOperation(), frozenRefCountPatterns);
  }
};

} // namespace
} // namespace trident::torch
