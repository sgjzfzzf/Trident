//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h> // NOLINT(misc-include-cleaner)
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <optional>
#include <string>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>
#include <type_traits>
#include <utility>

namespace trident::torch {
#define GEN_PASS_DEF_CONVERTTORCHTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"

static tvm_ffi::UnionType getStringUnionType(mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::Type> const stringTypes = {
      tvm_ffi::RawStrType::get(context), tvm_ffi::SmallStrType::get(context),
      tvm_ffi::StrType::get(context)};
  return tvm_ffi::UnionType::get(context, stringTypes);
}

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
                           mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<mlir::torch::Torch::OperatorOp>(typeConverter,
                                                                  ctx, 1),
        typeConverter(typeConverter) {}

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

template <typename ConstructOp>
class ConvertTorchArrayConstruct final
    : public mlir::OpConversionPattern<ConstructOp> {
public:
  explicit ConvertTorchArrayConstruct(
      const TorchFFITypeConverter &typeConverter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ConstructOp>(typeConverter, ctx, 1) {}

  mlir::LogicalResult
  matchAndRewrite(ConstructOp op, typename ConstructOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tvm_ffi::ArrayCreateOp>(
        op, tvm_ffi::ArrayType::get(this->getContext()), adaptor.getElements());
    return mlir::success();
  }
};

class ConvertTorchArrayUnpack final
    : public mlir::OpConversionPattern<mlir::torch::Torch::PrimListUnpackOp> {
public:
  explicit ConvertTorchArrayUnpack(const TorchFFITypeConverter &typeConverter,
                                   mlir::MLIRContext *ctx)
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
                                mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ConstantOp>(typeConverter, ctx, 1),
        typeConverter(typeConverter) {}

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
      rewriter.replaceOp(op, string.getResult(0));
      return mlir::success();
    }
    rewriter.replaceOpWithNewOp<tvm_ffi::ConstantOp>(op, targetType, value);
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
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

class ConvertTorchExtGet final
    : public mlir::OpConversionPattern<trident::torchext::GetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(trident::torchext::GetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tvm_ffi::GetOp>(op, rewriter.getI1Type(),
                                                adaptor.getOperand());
    return mlir::success();
  }
};

class ConvertTorchValueTensorLiteralOp final
    : public mlir::OpConversionPattern<
          mlir::torch::Torch::ValueTensorLiteralOp> {
public:
  using OpConversionPattern::OpConversionPattern;

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
    mlir::RankedTensorType const denseType =
        mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
    if (!denseType || denseType.getShape() != tensorType.getSizes()) {
      return op.emitError("literal shape does not match result type");
    }
    rewriter.replaceOpWithNewOp<tvm_ffi::TensorLiteralOp>(
        op, tvm_ffi::TensorType::get(rewriter.getContext()), dense);
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
        ConvertTorchConstant<mlir::torch::Torch::ConstantStrOp>>(typeConverter,
                                                                 &getContext());
    conversionPatterns.add<ConvertTorchExtConvert>(typeConverter,
                                                   &getContext());
    conversionPatterns.add<ConvertTorchExtGet>(typeConverter, &getContext());
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
    conversionPatterns.add<ConvertGenericOp<mlir::func::CallOp>,
                           ConvertGenericOp<tvm_ffi::ArrayCreateOp>,
                           ConvertGenericOp<tvm_ffi::CallOp>,
                           ConvertGenericOp<tvm_ffi::ConstantOp>,
                           ConvertGenericOp<tvm_ffi::ExceptionOp>,
                           ConvertGenericOp<tvm_ffi::FunctionCallOp>>(
        typeConverter, &getContext());

    mlir::ConversionTarget conversionTarget(getContext());
    conversionTarget
        .addLegalDialect<mlir::arith::ArithDialect, mlir::BuiltinDialect,
                         trident::arithext::ArithExtDialect>();
    conversionTarget.addLegalOp<
        tvm_ffi::ArrayCreateOp, tvm_ffi::CallOp, tvm_ffi::ConstantOp,
        tvm_ffi::ExceptionOp, tvm_ffi::FunctionCallOp,
        tvm_ffi::FunctionGetGlobalOp, tvm_ffi::GetOp, tvm_ffi::ObjectDecRefOp,
        tvm_ffi::ObjectIncRefOp, tvm_ffi::TensorLiteralOp>();
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
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, conversionPatterns, conversionTarget);
    conversionTarget.addDynamicallyLegalOp<
        tvm_ffi::ArrayGetItemOp, tvm_ffi::ArrayLengthOp, tvm_ffi::CastOp,
        tvm_ffi::EqOp, tvm_ffi::TensorDeviceOp, tvm_ffi::TensorDimOp,
        tvm_ffi::TensorDTypeOp, tvm_ffi::TensorSizeOp,
        tvm_ffi::TensorStorageOffsetOp, tvm_ffi::TensorStrideOp>(
        [&](mlir::Operation *op) { return typeConverter.isLegal(op); });
    conversionTarget.addLegalOp<mlir::ModuleOp, tvm_ffi::ReturnOp>();
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
  }
};

} // namespace trident::torch
