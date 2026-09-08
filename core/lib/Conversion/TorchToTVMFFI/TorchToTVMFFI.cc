//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchToTVMFFI/TorchToTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/Torch/IR/TorchInterfaces.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtInterfaces.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <cstdint>
#include <dlpack/dlpack.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <string>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h>
#include <utility>

namespace trident::conversion {
#include "TorchToTVMFFIPDLLPatterns.h.inc"

#define GEN_PASS_DEF_CONVERTTORCHTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"

void populateTorchToTVMFFITypeConversions(mlir::TypeConverter &typeConverter) {
  typeConverter.addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
    if (type.getDialect().getNamespace() != "torch" &&
        !mlir::isa<torchext::DTypeType>(type)) {
      return std::nullopt;
    }
    if (mlir::isa<trident::torch::TorchToTVMFFITypeInterface>(type)) {
      return mlir::cast<trident::torch::TorchToTVMFFITypeInterface>(type)
          .getTVMFFIType();
    }
    return tvm_ffi::AnyType::get(type.getContext());
  });
}

/// TypeConverter used by this bridge.  Keeping the Torch-to-semantic mapping
/// in a TypeConverter makes it reusable by conversion patterns and gives us
/// the standard MLIR materialization path for the temporary boundary casts.
class TorchFFITypeConverter final : public mlir::TypeConverter {
public:
  TorchFFITypeConverter() {
    addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
      return mlir::isa<tvm_ffi::TVMFFIABIType>(type)
                 ? std::optional<mlir::Type>(type)
                 : std::nullopt;
    });
    populateTorchToTVMFFITypeConversions(*this);
    addConversion([](mlir::IntegerType type) -> mlir::Type { return type; });
    addConversion([](mlir::FloatType type) -> mlir::Type { return type; });
    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      if (inputs.size() == 1 &&
          mlir::isa<tvm_ffi::TVMFFIABIType>(inputs.front().getType()) &&
          mlir::isa<tvm_ffi::AnyType, tvm_ffi::UnionType>(type)) {
        return tvm_ffi::CastOp::create(builder, loc, type, inputs.front());
      }
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

tvm_ffi::FunctionCallOp createCheckedFunctionCall(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Type resultType,
                                                  llvm::StringRef callee,
                                                  mlir::ValueRange arguments) {
  tvm_ffi::FunctionGetGlobalOp getGlobal = tvm_ffi::FunctionGetGlobalOp::create(
      builder, loc, tvm_ffi::FunctionType::get(builder.getContext()),
      builder.getI1Type(), callee);
  std::string const getGlobalError =
      llvm::formatv("TVMFFIFunctionGetGlobal failed for {0}", callee);
  mlir::cf::AssertOp::create(builder, loc, getGlobal.getSuccess(),
                             getGlobalError);

  tvm_ffi::FunctionCallOp call = tvm_ffi::FunctionCallOp::create(
      builder, loc, resultType, builder.getI1Type(), getGlobal.getResult(),
      arguments);
  std::string const callError =
      llvm::formatv("TVMFFIFunctionCall failed for {0}", callee);
  mlir::cf::AssertOp::create(builder, loc, call.getSuccess(), callError);
  return call;
}

class ConvertAtenLibraryCall final
    : public mlir::OpConversionPattern<mlir::torch::Torch::OperatorOp> {
public:
  explicit ConvertAtenLibraryCall(const TorchFFITypeConverter &typeConverter,
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
    if (op->getNumResults() == 0) {
      return op.emitError("TVM FFI function calls require one result");
    } else if (op->getNumResults() == 1) {
      mlir::Type const resultType =
          typeConverter.convertType(op->getResult(0).getType());
      if (!resultType) {
        return mlir::failure();
      }
      tvm_ffi::FunctionCallOp call = createCheckedFunctionCall(
          rewriter, op->getLoc(), resultType, callee, operands);
      replacements.push_back(call.getResult());
    } else {
      mlir::Type const arrayType = tvm_ffi::ArrayType::get(getContext());
      tvm_ffi::FunctionCallOp call = createCheckedFunctionCall(
          rewriter, op->getLoc(), arrayType, callee, operands);
      for (auto [index, result] : llvm::enumerate(op->getResults())) {
        tvm_ffi::ConstantIntOp idx = tvm_ffi::ConstantIntOp::create(
            rewriter, op->getLoc(), tvm_ffi::IntType::get(getContext()),
            rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
        mlir::Type const base = typeConverter.convertType(result.getType());
        if (!base) {
          return mlir::failure();
        }
        tvm_ffi::FunctionCallOp item = createCheckedFunctionCall(
            rewriter, op->getLoc(), base, "ffi.ArrayGetItem",
            {call.getResult(), idx.getResult()});
        replacements.push_back(item.getResult());
      }
    }
    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
};

class ConvertAtenLenT final
    : public mlir::OpConversionPattern<mlir::torch::Torch::OperatorOp> {
public:
  explicit ConvertAtenLenT(const TorchFFITypeConverter &typeConverter,
                           mlir::MLIRContext *context)
      : mlir::OpConversionPattern<mlir::torch::Torch::OperatorOp>(typeConverter,
                                                                  context, 2),
        typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::OperatorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.getName() != "torch.aten.len.t") {
      return mlir::failure();
    }
    if (op->getNumResults() != 1) {
      return op.emitError("expected exactly one result");
    }
    mlir::Type const resultType =
        typeConverter.convertType(op->getResult(0).getType());
    if (!resultType) {
      return op.emitError("cannot convert result type");
    }
    tvm_ffi::FunctionCallOp call =
        createCheckedFunctionCall(rewriter, op.getLoc(), resultType,
                                  "ffi.ArraySize", adaptor.getOperands());
    rewriter.replaceOp(op, call.getResult());
    return mlir::success();
  }

private:
  const TorchFFITypeConverter &typeConverter;
};

class ConvertTorchExtEq final
    : public mlir::OpConversionPattern<torchext::EqOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(torchext::EqOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tvm_ffi::EqOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return mlir::success();
  }
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

template <typename Op>
class ConvertTorchArrayConstruct final : public mlir::OpConversionPattern<Op> {
public:
  explicit ConvertTorchArrayConstruct(
      const TorchFFITypeConverter &typeConverter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<Op>(typeConverter, ctx, 1) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    tvm_ffi::FunctionCallOp call = createCheckedFunctionCall(
        rewriter, op.getLoc(), tvm_ffi::ArrayType::get(this->getContext()),
        "ffi.Array", adaptor.getElements());
    rewriter.replaceOp(op, call.getResult());
    return mlir::success();
  }
};

template <typename Op>
class ConvertTorchArrayUnpack final : public mlir::OpConversionPattern<Op> {
public:
  explicit ConvertTorchArrayUnpack(const TorchFFITypeConverter &typeConverter,
                                   mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<Op>(typeConverter, ctx, 1),
        typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value const array = adaptor.getOperands().front();
    llvm::SmallVector<mlir::Value> replacements;
    replacements.reserve(op.getNumResults());
    for (auto [index, result] : llvm::enumerate(op.getResults())) {
      mlir::Type const resultType = typeConverter.convertType(result.getType());
      if (!resultType) {
        return op.emitError("cannot convert unpack result type");
      }
      tvm_ffi::ConstantIntOp idx = tvm_ffi::ConstantIntOp::create(
          rewriter, op.getLoc(), tvm_ffi::IntType::get(this->getContext()),
          rewriter.getI64IntegerAttr(static_cast<int64_t>(index)));
      tvm_ffi::FunctionCallOp item = createCheckedFunctionCall(
          rewriter, op.getLoc(), resultType, "ffi.ArrayGetItem",
          {array, idx.getResult()});
      replacements.push_back(item.getResult());
    }
    rewriter.replaceOp(op, replacements);
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
    : public mlir::OpConversionPattern<torchext::ConvertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(torchext::ConvertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    tvm_ffi::FunctionCallOp call = createCheckedFunctionCall(
        rewriter, op.getLoc(), tvm_ffi::IntType::get(getContext()),
        "trident.runtime.tvm_ffi_to_torch_type", adaptor.getOperands());
    rewriter.replaceOp(op, call.getResult());
    return mlir::success();
  }
};

class ConvertTorchExtConstantDType final
    : public mlir::OpConversionPattern<torchext::ConstantDTypeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(torchext::ConstantDTypeOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dtype =
        mlir::cast<trident::torchext::DTypeAttrInterface>(op.getValue());
    DLDataType const dlpackDType = dtype.getDLPackDType();
    mlir::ArrayAttr const value = mlir::ArrayAttr::get(
        getContext(), {rewriter.getI64IntegerAttr(dlpackDType.code),
                       rewriter.getI64IntegerAttr(dlpackDType.bits),
                       rewriter.getI64IntegerAttr(dlpackDType.lanes)});
    rewriter.replaceOpWithNewOp<tvm_ffi::ConstantDTypeOp>(op, value);
    return mlir::success();
  }
};

class ConvertTorchExtGet final
    : public mlir::OpConversionPattern<torchext::GetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(torchext::GetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tvm_ffi::GetOp>(op, op.getResult().getType(),
                                                adaptor.getOperand());
    return mlir::success();
  }
};

template <typename Op>
class ConvertTorchConversionTo final : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tvm_ffi::GetOp>(op, op.getResult().getType(),
                                                adaptor.getOperand());
    return mlir::success();
  }
};

class ConvertTorchTensorGet final
    : public mlir::OpConversionPattern<tvm_ffi::GetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::GetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!mlir::isa<mlir::torch::Torch::BaseTensorType>(
            op.getOperand().getType())) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<tvm_ffi::GetOp>(op, op.getResult().getType(),
                                                adaptor.getOperand());
    return mlir::success();
  }
};

template <typename Op, typename TargetOp>
class ConvertTorchExtTensorMetadata final
    : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, op->getResultTypes(),
                                          adaptor.getTensor());
    return mlir::success();
  }
};

template <typename Op, typename TargetOp>
class ConvertTorchExtTensorIndexedMetadata final
    : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(
        op, op->getResultTypes(), adaptor.getTensor(), adaptor.getIndex());
    return mlir::success();
  }
};

/// Lower a value-semantic copy to an explicit TVM FFI tensor clone.  The
/// operation owns a newly allocated storage and preserves the input layout.
class ConvertTorchCopyToValueTensor final
    : public mlir::OpConversionPattern<
          mlir::torch::Torch::CopyToValueTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::CopyToValueTensorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tvm_ffi::TensorCloneOp>(
        op, tvm_ffi::TensorType::get(rewriter.getContext()),
        adaptor.getOperand());
    return mlir::success();
  }
};

/// Lower the mutation marker to an explicit TVM FFI in-place copy operation.
class ConvertTorchOverwriteTensorContents final
    : public mlir::OpConversionPattern<
          mlir::torch::Torch::OverwriteTensorContentsOp> {
public:
  ConvertTorchOverwriteTensorContents(
      const TorchFFITypeConverter &typeConverter, mlir::MLIRContext *context)
      : mlir::OpConversionPattern<
            mlir::torch::Torch::OverwriteTensorContentsOp>(typeConverter,
                                                           context, 100) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::torch::Torch::OverwriteTensorContentsOp op,
                  OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tvm_ffi::TensorCopyOp>(
        op, adaptor.getOverwritten(), adaptor.getValue());
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
    : public mlir::OpConversionPattern<tvm_ffi::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tvm_ffi::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<tvm_ffi::ReturnOp>(op, adaptor.getOperands());
    return mlir::success();
  }
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

    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);
    mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<tvm_ffi::FuncOp>(
        patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateGeneratedPDLLPatterns(patterns,
                                  mlir::PDLConversionConfig(&typeConverter));
    patterns.add<
        ConvertAtenLenT, ConvertAtenLibraryCall,
        ConvertTorchArrayConstruct<mlir::torch::Torch::PrimListConstructOp>,
        ConvertTorchArrayConstruct<mlir::torch::Torch::PrimTupleConstructOp>,
        ConvertTorchArrayUnpack<mlir::torch::Torch::PrimListUnpackOp>,
        ConvertTorchArrayUnpack<mlir::torch::Torch::PrimTupleUnpackOp>,
        ConvertTorchConversionTo<mlir::torch::TorchConversion::ToF64Op>,
        ConvertTorchConversionTo<mlir::torch::TorchConversion::ToI1Op>,
        ConvertTorchConversionTo<mlir::torch::TorchConversion::ToI64Op>,
        ConvertTorchCopyToValueTensor, ConvertTorchExtConstantDType,
        ConvertTorchExtConvert, ConvertTorchExtEq, ConvertTorchExtGet,
        ConvertTorchExtTensorMetadata<torchext::TensorDeviceOp,
                                      tvm_ffi::TensorDeviceOp>,
        ConvertTorchExtTensorMetadata<torchext::TensorDimOp,
                                      tvm_ffi::TensorDimOp>,
        ConvertTorchExtTensorMetadata<torchext::TensorDTypeOp,
                                      tvm_ffi::TensorDTypeOp>,
        ConvertTorchExtTensorMetadata<torchext::TensorStorageOffsetOp,
                                      tvm_ffi::TensorStorageOffsetOp>,
        ConvertTorchExtTensorIndexedMetadata<torchext::TensorSizeOp,
                                             tvm_ffi::TensorSizeOp>,
        ConvertTorchExtTensorIndexedMetadata<torchext::TensorStrideOp,
                                             tvm_ffi::TensorStrideOp>,
        ConvertTorchOverwriteTensorContents, ConvertTorchValueTensorLiteralOp,
        ConvertTVMFFIReturn, ConvertGenericOp<tvm_ffi::CastOp>,
        ConvertGenericOp<tvm_ffi::EqOp>,
        ConvertGenericOp<tvm_ffi::TensorDeviceOp>,
        ConvertGenericOp<tvm_ffi::TensorDimOp>,
        ConvertGenericOp<tvm_ffi::TensorDTypeOp>,
        ConvertGenericOp<tvm_ffi::TensorSizeOp>,
        ConvertGenericOp<tvm_ffi::TensorStorageOffsetOp>,
        ConvertGenericOp<tvm_ffi::TensorStrideOp>,
        ConvertGenericOp<mlir::func::CallOp>, ConvertGenericOp<tvm_ffi::CallOp>,
        ConvertGenericOp<tvm_ffi::ExceptionOp>,
        ConvertGenericOp<tvm_ffi::FunctionCallOp>>(typeConverter,
                                                   &getContext());

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::BuiltinDialect,
                           mlir::LLVM::LLVMDialect>();
    target.addLegalOp<tvm_ffi::CallOp, tvm_ffi::ConstantBoolOp,
                      tvm_ffi::ConstantDeviceOp, tvm_ffi::ConstantDTypeOp,
                      tvm_ffi::ConstantFloatOp, tvm_ffi::ConstantIntOp,
                      tvm_ffi::ConstantNoneOp, tvm_ffi::ConstantRawStrOp,
                      tvm_ffi::ExceptionOp, tvm_ffi::FunctionCallOp,
                      tvm_ffi::FunctionGetGlobalOp, tvm_ffi::ObjectDecRefOp,
                      tvm_ffi::ObjectIncRefOp, tvm_ffi::TensorCloneOp,
                      tvm_ffi::TensorCopyOp, tvm_ffi::TensorLiteralOp,
                      tvm_ffi::ToOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp func) -> bool {
          return typeConverter.isSignatureLegal(func.getFunctionType());
        });
    target.markOpRecursivelyLegal<mlir::func::FuncOp>(
        [](mlir::func::FuncOp) -> bool { return false; });
    target.addDynamicallyLegalOp<tvm_ffi::FuncOp>(
        [&](tvm_ffi::FuncOp func) -> bool {
          return typeConverter.isSignatureLegal(func.getFunctionType());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp ret) -> bool {
          return mlir::isLegalForReturnOpTypeConversionPattern(ret,
                                                               typeConverter);
        });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp call) -> bool {
          return llvm::all_of(call.getOperandTypes(),
                              [&](mlir::Type type) -> bool {
                                return typeConverter.isLegal(type);
                              }) &&
                 llvm::all_of(call.getResultTypes(),
                              [&](mlir::Type type) -> bool {
                                return typeConverter.isLegal(type);
                              });
        });
    mlir::cf::populateCFStructuralTypeConversionsAndLegality(typeConverter,
                                                             patterns, target);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);
    target.addDynamicallyLegalOp<tvm_ffi::CastOp, tvm_ffi::EqOp,
                                 tvm_ffi::TensorDeviceOp, tvm_ffi::TensorDimOp,
                                 tvm_ffi::TensorDTypeOp, tvm_ffi::TensorSizeOp,
                                 tvm_ffi::TensorStorageOffsetOp,
                                 tvm_ffi::TensorStrideOp, tvm_ffi::GetOp>(
        [&](mlir::Operation *op) -> bool { return typeConverter.isLegal(op); });
    target.addLegalOp<mlir::ModuleOp, mlir::cf::AssertOp>();
    target.addDynamicallyLegalOp<tvm_ffi::ReturnOp>(
        [&](tvm_ffi::ReturnOp op) -> bool {
          return typeConverter.isLegal(op);
        });
    // Torch, TorchConversion, and TorchExt operations must all be lowered
    // before this conversion completes. In particular, a Triton kernel launch
    // must have been converted to gpu.launch_func by ConvertTorchExtToGPU.
    target
        .addIllegalDialect<mlir::torch::Torch::TorchDialect,
                           mlir::torch::TorchConversion::TorchConversionDialect,
                           torchext::TorchExtDialect>();
    patterns.add<ConvertTorchTensorGet,
                 MaterializeTorchExtOperands<torchext::GetOp>>(typeConverter,
                                                               &getContext());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace trident::conversion
