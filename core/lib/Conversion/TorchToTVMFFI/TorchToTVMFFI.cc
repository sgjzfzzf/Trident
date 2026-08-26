//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "dlpack/dlpack.h"
#include "trident/core/Conversion/Utils/AOTICAPIDescriptors.h"
#include "trident/core/Conversion/Utils/Check.h"
#include "trident/core/Conversion/Utils/TVMFFIUtils.h"
#include "trident/core/Conversion/Utils/Type.h"
#include "trident/core/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <optional>
#include <string>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>
#include <type_traits>
#include <utility>

namespace trident::torch {
#define GEN_PASS_DEF_CONVERTTORCHTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"

namespace {

using OwnedValues =
    llvm::DenseMap<mlir::Region *, llvm::SmallSetVector<mlir::Value, 4>>;

static tvm_ffi::UnionType getStringUnionType(mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::Type> const stringTypes = {
      tvm_ffi::RawStrType::get(context), tvm_ffi::SmallStrType::get(context),
      tvm_ffi::StrType::get(context)};
  return tvm_ffi::UnionType::get(context, stringTypes);
}

} // namespace

mlir::Type convertTorchTypeToTVMFFIType(mlir::Type type) {
  mlir::MLIRContext *context = type.getContext();
  return llvm::TypeSwitch<mlir::Type, mlir::Type>(type)
      .Case<mlir::torch::Torch::AnyType>(
          [&](mlir::Type) { return tvm_ffi::AnyType::get(context); })
      .Case<mlir::torch::Torch::BoolType>(
          [&](mlir::Type) { return tvm_ffi::BoolType::get(context); })
      .Case<mlir::torch::Torch::DeviceType>(
          [&](mlir::Type) { return tvm_ffi::DeviceType::get(context); })
      .Case<trident::torchext::DTypeType>(
          [&](mlir::Type) { return tvm_ffi::DTypeType::get(context); })
      .Case<mlir::torch::Torch::FloatType>(
          [&](mlir::Type) { return tvm_ffi::FloatType::get(context); })
      .Case<mlir::torch::Torch::IntType>(
          [&](mlir::Type) { return tvm_ffi::IntType::get(context); })
      .Case<mlir::torch::Torch::ListType, mlir::torch::Torch::TupleType>(
          [&](mlir::Type) { return tvm_ffi::ArrayType::get(context); })
      .Case<mlir::torch::Torch::NoneType>(
          [&](mlir::Type) { return tvm_ffi::NoneType::get(context); })
      .Case<mlir::torch::Torch::NonValueTensorType,
            mlir::torch::Torch::ValueTensorType>(
          [&](mlir::Type) { return tvm_ffi::TensorType::get(context); })
      .Case<mlir::torch::Torch::StringType>(
          [&](mlir::Type) { return getStringUnionType(context); })
      .Default([&](mlir::Type) { return tvm_ffi::AnyType::get(context); });
}

void populateTorchToTVMFFITypeConversions(mlir::TypeConverter &typeConverter) {
  typeConverter.addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
    if (type.getDialect().getNamespace() != "torch" &&
        !mlir::isa<trident::torchext::DTypeType>(type)) {
      return std::nullopt;
    }
    return convertTorchTypeToTVMFFIType(type);
  });
}

namespace {

static void recordOwnedObjectResults(mlir::Operation *operation,
                                     mlir::ValueRange results,
                                     OwnedValues &ownedValues) {
  for (mlir::Value const result : results) {
    if (mlir::Type type = result.getType();
        mlir::isa<tvm_ffi::AnyType, tvm_ffi::UnionType>(type) ||
        type.hasTrait<mlir::TypeTrait::Object>()) {
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
      return type.hasTrait<mlir::TypeTrait::TVMFFIABI>()
                 ? std::optional<mlir::Type>(type)
                 : std::nullopt;
    });
    populateTorchToTVMFFITypeConversions(*this);
    addConversion([](mlir::IntegerType type) -> mlir::Type { return type; });
    addConversion([](mlir::FloatType type) -> mlir::Type { return type; });
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
  mlir::Location const loc = terminator->getLoc();
  llvm::SmallSetVector<mlir::Value, 4> valuesToRelease(ownedValues);

  for (mlir::Value const operand : terminator->getOperands()) {
    mlir::Value value = operand;
    mlir::UnrealizedConversionCastOp cast;
    while ((cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) &&
           cast->getNumOperands() == 1) {
      value = cast->getOperand(0);
    }
    mlir::Type convertedType = typeConverter.convertType(value.getType());
    if (convertedType &&
        (mlir::isa<tvm_ffi::AnyType, tvm_ffi::UnionType>(convertedType) ||
         convertedType.hasTrait<mlir::TypeTrait::Object>())) {
      if (value.getType() != convertedType) {
        value = typeConverter.materializeTargetConversion(builder, loc,
                                                          convertedType, value);
        if (!value) {
          continue;
        }
      }
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

  for (mlir::Value const value : valuesToRelease) {
    mlir::Type const convertedType = typeConverter.convertType(value.getType());
    if (!convertedType || value.getType() == convertedType) {
      tvm_ffi::ObjectDecRefOp::create(builder, loc, value);
    } else {
      mlir::Value const convertedValue =
          typeConverter.materializeTargetConversion(builder, loc, convertedType,
                                                    value);
      if (convertedValue) {
        tvm_ffi::ObjectDecRefOp::create(builder, loc, convertedValue);
      }
    }
  }
}

class ConvertArrayGetItem final
    : public mlir::OpConversionPattern<tvm_ffi::ArrayGetItemOp> {
public:
  explicit ConvertArrayGetItem(const TorchFFITypeConverter &typeConverter,
                               mlir::MLIRContext *context)
      : mlir::OpConversionPattern<tvm_ffi::ArrayGetItemOp>(typeConverter,
                                                           context),
        typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ArrayGetItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type const resultType =
        typeConverter.convertType(op.getResult().getType());
    if (!resultType) {
      return op.emitError("cannot convert array element type");
    }
    rewriter.replaceOpWithNewOp<tvm_ffi::ArrayGetItemOp>(
        op, resultType, adaptor.getArray(), adaptor.getIndex(),
        mlir::TypeAttr::get(resultType));
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
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
    llvm::StringRef const name = op.getName();
    if (!name.starts_with("torch.aten.")) {
      return mlir::failure();
    }
    std::string const callee =
        ("trident." + name.drop_front(sizeof("torch.") - 1)).str();
    mlir::ValueRange const operands = adaptor.getOperands();
    llvm::SmallVector<mlir::Value> replacements;
    if (op->getNumResults() <= 1) {
      llvm::SmallVector<mlir::Type> resultTypes;
      if (op->getNumResults()) {
        mlir::Type const resultType =
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
      mlir::Type const arrayType = tvm_ffi::ArrayType::get(getContext());
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
            rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
        mlir::Type const base = typeConverter.convertType(result.getType());
        if (!base) {
          return mlir::failure();
        }
        mlir::Type const semantic = base;

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
    if (auto it = ownedValues.find(region); it != ownedValues.end()) {
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
    tvm_ffi::ArrayCreateOp const array =
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
    mlir::Value const array = adaptor.getOperand();
    llvm::SmallVector<mlir::Value> replacements;
    replacements.reserve(op.getNumResults());
    for (auto [index, result] : llvm::enumerate(op.getResults())) {
      mlir::Type const resultType = typeConverter.convertType(result.getType());
      if (!resultType) {
        return op.emitError("cannot convert ListUnpack result type");
      }
      tvm_ffi::ConstantOp idx = tvm_ffi::ConstantOp::create(
          rewriter, op.getLoc(), tvm_ffi::IntType::get(getContext()),
          rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
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
    mlir::Type targetType;
    if (mlir::isa<mlir::torch::Torch::StringType>(op.getResult().getType())) {
      targetType = tvm_ffi::RawStrType::get(op.getContext());
    } else {
      targetType = typeConverter.convertType(op.getResult().getType());
    }
    if (!targetType) {
      return mlir::failure();
    }
    mlir::Attribute value = // NOLINT(misc-const-correctness)
        op->getAttr("value");
    if constexpr (std::is_same_v<ConstantOp,
                                 mlir::torch::Torch::ConstantNoneOp>) {
      value = rewriter.getUnitAttr();
    }
    if (!value) {
      return op->emitError("constant is missing value attribute");
    }
    if constexpr (std::is_same_v<ConstantOp,
                                 mlir::torch::Torch::ConstantStrOp>) {
      tvm_ffi::ConstantOp raw = tvm_ffi::ConstantOp::create(
          rewriter, op.getLoc(), tvm_ffi::RawStrType::get(op.getContext()),
          value);
      tvm_ffi::FunctionGetGlobalOp getGlobal =
          tvm_ffi::FunctionGetGlobalOp::create(
              rewriter, op.getLoc(),
              tvm_ffi::FunctionType::get(op.getContext()), "ffi.String");
      tvm_ffi::FunctionCallOp string = tvm_ffi::FunctionCallOp::create(
          rewriter, op.getLoc(),
          mlir::TypeRange{getStringUnionType(op.getContext())},
          getGlobal.getResult(), mlir::ValueRange{raw.getResult()});
      recordOwnedObjectResults(string, string->getResults(), ownedValues);
      rewriter.replaceOp(op, string.getResult(0));
      return mlir::success();
    }
    tvm_ffi::ConstantOp const constant =
        rewriter.replaceOpWithNewOp<tvm_ffi::ConstantOp>(op, targetType, value);
    recordOwnedObjectResults(constant, constant->getResults(), ownedValues);
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
  llvm::DenseMap<mlir::Region *, llvm::SmallSetVector<mlir::Value, 4>>
      &ownedValues;
};

/// Convert a TorchExt dtype wrapper to the TVM FFI integer consumed by Torch
/// operations after this pass.  This pattern intentionally lives in the same
/// conversion as the Torch users so the producer and its users agree on the
/// converted result type.
class ConvertTorchExtConvert final
    : public mlir::OpConversionPattern<trident::torchext::ConvertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(trident::torchext::ConvertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    tvm_ffi::FunctionGetGlobalOp getGlobal =
        tvm_ffi::FunctionGetGlobalOp::create(
            rewriter, op.getLoc(), tvm_ffi::FunctionType::get(getContext()),
            "trident.runtime.tvm_ffi_to_torch_type");
    rewriter.replaceOpWithNewOp<tvm_ffi::FunctionCallOp>(
        op, mlir::TypeRange{tvm_ffi::IntType::get(getContext())},
        getGlobal.getResult(), adaptor.getOperands());
    return mlir::success();
  }
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
    mlir::DenseElementsAttr const dense =
        mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    mlir::torch::Torch::BaseTensorType const tensorType =
        mlir::dyn_cast<mlir::torch::Torch::BaseTensorType>(op.getType());
    if (!dense || !tensorType || !tensorType.hasSizes()) {
      return op.emitError("literal requires a dense tensor and static shape");
    }
    llvm::ArrayRef<int64_t> const shape = tensorType.getSizes();
    mlir::RankedTensorType const denseType =
        mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
    if (!denseType || denseType.getShape() != shape) {
      return op.emitError("literal shape does not match result type");
    }

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::ModuleOp module = op->template getParentOfType<mlir::ModuleOp>();
    mlir::LLVM::LLVMPointerType ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::IntegerType const i8Ty = mlir::IntegerType::get(ctx, 8);
    mlir::IntegerType const i32Ty = mlir::IntegerType::get(ctx, 32);
    mlir::IntegerType const i64Ty = mlir::IntegerType::get(ctx, 64);
    mlir::LLVM::LLVMStructType const anyTy =
        conversion::utils::getTVMFFIAnyType(ctx);

    int64_t const elementCount = dense.getNumElements();
    int64_t elementBits = 0;
    int32_t dtypeCode = 0;
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
    mlir::Value const dtypeArg = conversion::utils::buildDTypeAnySlot(
        rewriter, op.getLoc(), dtypeCode, elementBits);
    mlir::FailureOr<mlir::Value> dtypeResult =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, op.getLoc(), module,
            "trident.runtime.tvm_ffi_to_torch_type", {dtypeArg});
    if (mlir::failed(dtypeResult)) {
      return op.emitError("failed to call TVM FFI dtype conversion helper");
    }
    mlir::Value const dtype = conversion::utils::loadIntFromAnySlot(
        rewriter, op.getLoc(), dtypeResult.value());
    mlir::Value const rank = mlir::LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Ty, static_cast<int64_t>(shape.size()));
    mlir::Value const sizes = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                       static_cast<int64_t>(shape.size())));
    mlir::Value const strides = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, i64Ty,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                       static_cast<int64_t>(shape.size())));
    int64_t stride = 1;
    for (int64_t index = static_cast<int64_t>(shape.size()) - 1; index >= 0;
         --index) {
      mlir::Value const sizeSlot = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), ptrTy, i64Ty, sizes,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::Value const strideSlot = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), ptrTy, i64Ty, strides,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::LLVM::StoreOp::create(
          rewriter, op.getLoc(),
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                         static_cast<int64_t>(shape[index])),
          sizeSlot);
      mlir::LLVM::StoreOp::create(
          rewriter, op.getLoc(),
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, stride),
          strideSlot);
      stride *= shape[index];
    }

    mlir::Type storageType = denseType.getElementType();
    if (mlir::IntegerType const integerType =
            mlir::dyn_cast<mlir::IntegerType>(storageType);
        integerType && integerType.isSignlessInteger(1)) {
      storageType = i8Ty;
    }
    mlir::Value data = mlir::LLVM::AllocaOp::create(
        rewriter, op.getLoc(), ptrTy, storageType,
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty,
                                       elementCount));
    auto storeElement = [&](int64_t index, mlir::Value value) {
      mlir::Value const slot = mlir::LLVM::GEPOp::create(
          rewriter, op.getLoc(), ptrTy, storageType, data,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(index)});
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), value, slot);
    };
    if (mlir::isa<mlir::FloatType>(storageType)) {
      auto values = dense.getValues<llvm::APFloat>();
      for (auto [index, value] : llvm::enumerate(values)) {
        storeElement(static_cast<int64_t>(index),
                     mlir::LLVM::ConstantOp::create(
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
        storeElement(static_cast<int64_t>(index),
                     mlir::LLVM::ConstantOp::create(
                         rewriter, op.getLoc(), storageType,
                         mlir::IntegerAttr::get(storageType, value)));
      }
    }

    mlir::Value tensorHandle;
    mlir::Value const zeroI32 =
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, 0);
    mlir::Value const oneI64 =
        mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 1);
    auto callDeviceType =
        [&](mlir::Value deviceType) -> mlir::FailureOr<mlir::Value> {
      mlir::Value const argument =
          conversion::utils::buildIntAnySlot(rewriter, op.getLoc(), deviceType);
      mlir::FailureOr<mlir::Value> result =
          conversion::utils::callTVMFFIGlobalFunction(
              rewriter, op.getLoc(), module,
              "trident.runtime.tvm_ffi_device_to_torch_device_type",
              {argument});
      if (mlir::failed(result)) {
        return mlir::failure();
      }
      return conversion::utils::loadIntFromAnySlot(rewriter, op.getLoc(),
                                                   result.value());
    };

    if (dense.isSplat()) {
      double fillValue;
      if (mlir::isa<mlir::FloatType>(storageType)) {
        fillValue = dense.getSplatValue<llvm::APFloat>().convertToDouble();
      } else {
        llvm::APInt const value = dense.getSplatValue<llvm::APInt>();
        mlir::IntegerType const integerType =
            mlir::cast<mlir::IntegerType>(storageType);
        fillValue = integerType.isSignlessInteger(1)
                        ? static_cast<double>(value.getBoolValue())
                        : (integerType.isUnsignedInteger()
                               ? static_cast<double>(value.getZExtValue())
                               : static_cast<double>(value.getSExtValue()));
      }
      mlir::LLVM::LLVMFuncOp const fullFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchAtenFull(module));
      mlir::LLVM::LLVMFuncOp const getDeviceIndexFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(module));
      mlir::Value const cudaDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, kDLCUDA);
      mlir::Value const cudaDeviceType =
          TRIDENT_CHECK_FAILURE(callDeviceType(cudaDLDeviceType));
      mlir::Value const dtypeSlot = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, i32Ty, oneI64);
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), dtype, dtypeSlot);
      mlir::Value const deviceSlot = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, i32Ty, oneI64);
      mlir::LLVM::StoreOp::create(rewriter, op.getLoc(), cudaDeviceType,
                                  deviceSlot);
      mlir::Value const deviceIndexSlot = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, i32Ty, oneI64);
      mlir::LLVM::CallOp::create(rewriter, op.getLoc(), getDeviceIndexFn,
                                 deviceIndexSlot);
      mlir::Value const deviceIndex = mlir::LLVM::LoadOp::create(
          rewriter, op.getLoc(), i32Ty, deviceIndexSlot);
      mlir::Value const output = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, ptrTy, oneI64);
      mlir::Value const fill = mlir::LLVM::ConstantOp::create(
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
      mlir::LLVM::LLVMFuncOp const createFromBlobFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchCreateTensorFromBlob(module));
      mlir::LLVM::LLVMFuncOp const emptyStridedFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchEmptyStrided(module));
      mlir::LLVM::LLVMFuncOp const copyFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchCopy_(module));
      mlir::LLVM::LLVMFuncOp const deleteTensorFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchDeleteTensorObject(module));
      mlir::LLVM::LLVMFuncOp const getDeviceIndexFn = TRIDENT_CHECK_FAILURE(
          conversion::utils::getOrCreateAOTITorchGetCurrentDeviceIndex(module));
      mlir::Value const cpuDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, kDLCPU);
      mlir::Value const cudaDLDeviceType =
          mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i32Ty, kDLCUDA);
      mlir::Value const cpuDeviceType =
          TRIDENT_CHECK_FAILURE(callDeviceType(cpuDLDeviceType));
      mlir::Value const cudaDeviceType =
          TRIDENT_CHECK_FAILURE(callDeviceType(cudaDLDeviceType));
      mlir::Value const deviceIndexSlot = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, i32Ty, oneI64);
      mlir::LLVM::CallOp::create(rewriter, op.getLoc(), getDeviceIndexFn,
                                 deviceIndexSlot);
      mlir::Value const deviceIndex = mlir::LLVM::LoadOp::create(
          rewriter, op.getLoc(), i32Ty, deviceIndexSlot);
      mlir::Value const cpuOutput = mlir::LLVM::AllocaOp::create(
          rewriter, op.getLoc(), ptrTy, ptrTy, oneI64);
      mlir::LLVM::CallOp::create(
          rewriter, op.getLoc(), createFromBlobFn,
          {data, rank, sizes, strides,
           mlir::LLVM::ConstantOp::create(rewriter, op.getLoc(), i64Ty, 0),
           dtype, cpuDeviceType, zeroI32, cpuOutput});
      mlir::Value const cpuTensor =
          mlir::LLVM::LoadOp::create(rewriter, op.getLoc(), ptrTy, cpuOutput);
      mlir::Value const cudaOutput = mlir::LLVM::AllocaOp::create(
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

    mlir::Value const tensorArg = conversion::utils::buildOpaquePtrAnySlot(
        rewriter, op.getLoc(), tensorHandle);
    mlir::FailureOr<mlir::Value> resultSlot =
        conversion::utils::callTVMFFIGlobalFunction(
            rewriter, op.getLoc(), module,
            "trident.runtime.tensor_to_tvm_ffi_object", {tensorArg});
    if (mlir::failed(resultSlot)) {
      return op.emitError("failed to call TVM FFI tensor conversion helper");
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, anyTy,
                                                    resultSlot.value());
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
    mlir::FunctionType const type = func.getFunctionType();
    if (type.getNumResults() == 1 &&
        mlir::isa<tvm_ffi::AnyType, tvm_ffi::UnionType>(type.getResult(0))) {
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

template <typename OpType>
class MaterializeTorchExtOperands final
    : public mlir::OpConversionPattern<OpType> {
public:
  MaterializeTorchExtOperands(const TorchFFITypeConverter &typeConverter,
                              mlir::MLIRContext *context)
      : mlir::OpConversionPattern<OpType>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return mlir::success();
  }
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
    conversionPatterns.add<ConvertArrayGetItem>(typeConverter, &getContext());
    conversionPatterns.add<
        ConvertAtenCall,
        ConvertTorchArrayConstruct<mlir::torch::Torch::PrimListConstructOp>,
        ConvertTorchArrayConstruct<mlir::torch::Torch::PrimTupleConstructOp>,
        ConvertTorchArrayUnpack,
        ConvertTorchConstant<mlir::torch::Torch::ConstantBoolOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantDeviceOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantFloatOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantIntOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantNoneOp>,
        ConvertTorchConstant<mlir::torch::Torch::ConstantStrOp>>(
        typeConverter, ownedValues, &getContext());
    conversionPatterns.add<ConvertTorchExtConvert>(typeConverter,
                                                   &getContext());
    conversionPatterns.add<ConvertTorchValueTensorLiteralOp>(typeConverter,
                                                             &getContext());
    conversionPatterns
        .add<ConvertGenericOp<tvm_ffi::ArrayLengthOp>,
             ConvertGenericOp<tvm_ffi::CastOp>, ConvertGenericOp<tvm_ffi::EqOp>,
             ConvertGenericOp<tvm_ffi::TensorDeviceOp>,
             ConvertGenericOp<tvm_ffi::TensorDimOp>,
             ConvertGenericOp<tvm_ffi::TensorDTypeOp>,
             ConvertGenericOp<tvm_ffi::TensorSizeOp>,
             ConvertGenericOp<tvm_ffi::TensorStorageOffsetOp>,
             ConvertGenericOp<tvm_ffi::TensorStrideOp>>(typeConverter,
                                                        &getContext());

    mlir::ConversionTarget conversionTarget(getContext());
    conversionTarget.addLegalDialect<
        mlir::arith::ArithDialect, mlir::BuiltinDialect,
        mlir::LLVM::LLVMDialect, trident::arithext::ArithExtDialect>();
    conversionTarget
        .addLegalOp<tvm_ffi::ArrayCreateOp, tvm_ffi::CallOp,
                    tvm_ffi::ConstantOp, tvm_ffi::ExceptionOp,
                    tvm_ffi::FunctionCallOp, tvm_ffi::FunctionGetGlobalOp,
                    tvm_ffi::ObjectDecRefOp, tvm_ffi::ObjectIncRefOp>();
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
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(
        typeConverter, conversionPatterns, conversionTarget);
    conversionTarget.addDynamicallyLegalOp<
        tvm_ffi::ArrayGetItemOp, tvm_ffi::ArrayLengthOp, tvm_ffi::CastOp,
        tvm_ffi::EqOp, tvm_ffi::TensorDeviceOp, tvm_ffi::TensorDimOp,
        tvm_ffi::TensorDTypeOp, tvm_ffi::TensorSizeOp,
        tvm_ffi::TensorStorageOffsetOp, tvm_ffi::TensorStrideOp>(
        [&](mlir::Operation *op) { return typeConverter.isLegal(op); });
    conversionTarget
        .addLegalOp<mlir::ModuleOp, mlir::torch::Torch::PrimIfOp,
                    mlir::torch::Torch::PrimIfYieldOp, tvm_ffi::ReturnOp>();
    conversionTarget.addDynamicallyLegalOp<mlir::torch::Torch::OperatorOp>(
        [](mlir::torch::Torch::OperatorOp op) {
          return !op.getName().starts_with("torch.aten.");
        });
    conversionTarget.addIllegalOp<mlir::torch::Torch::ValueTensorLiteralOp>();
    conversionTarget.addDynamicallyLegalOp<
        trident::torchext::CastOp, trident::torchext::TridentKernelLaunchOp>(
        [&](mlir::Operation *op) { return typeConverter.isLegal(op); });
    conversionPatterns.add<
        MaterializeTorchExtOperands<trident::torchext::CastOp>,
        MaterializeTorchExtOperands<trident::torchext::TridentKernelLaunchOp>>(
        typeConverter, &getContext());
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
    refCountPatterns.add<ConvertTerminatorOp<mlir::func::ReturnOp>,
                         ConvertTerminatorOp<mlir::torch::Torch::PrimIfYieldOp>,
                         ConvertTerminatorOp<tvm_ffi::ReturnOp>>(
        &getContext(), ownedValues, typeConverter);
    mlir::FrozenRewritePatternSet const frozenRefCountPatterns(
        std::move(refCountPatterns));
    mlir::walkAndApplyPatterns(getOperation(), frozenRefCountPatterns);
  }
};

} // namespace
} // namespace trident::torch
