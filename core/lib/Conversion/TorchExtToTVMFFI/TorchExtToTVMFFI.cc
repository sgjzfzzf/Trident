//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchExtToTVMFFI/TorchExtToTVMFFI.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <torch-mlir/Dialect/Torch/IR/TorchTypes.h>
#include <utility>

namespace trident::torchext {

#define GEN_PASS_DEF_CONVERTTORCHEXTTOTVMFFI
#include "trident/core/Conversion/Passes.h.inc"

namespace {

class TorchExtToTVMFFITypeConverter : public mlir::TypeConverter {
public:
  TorchExtToTVMFFITypeConverter() {
    addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
      mlir::MLIRContext *context = type.getContext();
      if (mlir::isa<tvm_ffi::AnyType, tvm_ffi::ArrayType, tvm_ffi::BoolType,
                    tvm_ffi::DeviceType, tvm_ffi::DTypeType,
                    tvm_ffi::ExceptionType, tvm_ffi::FloatType,
                    tvm_ffi::IntType, tvm_ffi::NoneType, tvm_ffi::TensorType>(
              type)) {
        return type;
      }
      if (mlir::isa<DTypeType>(type)) {
        return tvm_ffi::DTypeType::get(context);
      }
      if (mlir::isa<mlir::torch::Torch::IntType>(type)) {
        return tvm_ffi::IntType::get(context);
      }
      return std::nullopt;
    });
    addConversion([](mlir::IntegerType type) -> mlir::Type { return type; });
    addConversion([](mlir::FloatType type) -> mlir::Type { return type; });
    addTargetMaterialization([](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs,
                                mlir::Location loc) -> mlir::Value {
      return mlir::UnrealizedConversionCastOp::create(
                 builder, loc, mlir::TypeRange(type), inputs)
          .getResult(0);
    });
  }
};

/// Lowers the TorchExt dtype wrapper to a TVM FFI scalar conversion call.
class ConvertTorchExtConvert final
    : public mlir::OpConversionPattern<ConvertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return op.emitError("failed to convert TorchExt conversion result type");
    }
    tvm_ffi::FunctionGetGlobalOp getGlobal =
        tvm_ffi::FunctionGetGlobalOp::create(
            rewriter, op.getLoc(), tvm_ffi::FunctionType::get(getContext()),
            "trident.runtime.tvm_ffi_to_torch_type");
    rewriter.replaceOpWithNewOp<tvm_ffi::FunctionCallOp>(
        op, mlir::TypeRange{resultType}, getGlobal.getResult(),
        adaptor.getOperands());
    return mlir::success();
  }
};

class ConvertTorchExtToTVMFFIPass final
    : public impl::ConvertTorchExtToTVMFFIBase<ConvertTorchExtToTVMFFIPass> {
  void runOnOperation() final {
    TorchExtToTVMFFITypeConverter typeConverter;
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalOp<tvm_ffi::FunctionGetGlobalOp, tvm_ffi::FunctionCallOp>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addIllegalOp<ConvertOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTorchExtConvert>(typeConverter, &getContext());
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace trident::torchext
