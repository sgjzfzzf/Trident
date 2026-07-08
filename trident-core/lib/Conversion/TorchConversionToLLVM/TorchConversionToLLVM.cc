//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-core/Conversion/TorchConversionToLLVM/TorchConversionToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "trident-core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

namespace trident::torch {

#define GEN_PASS_DEF_CONVERTTORCHCONVERSIONTOLLVM
#include "trident-core/Conversion/Passes.h.inc"

namespace {

/// Generic conversion pattern for TorchConversion materialization ops.
///
/// Each of these ops converts between a Torch dialect type and a builtin/LLVM
/// type (e.g. !torch.bool <-> i1, !torch.int <-> i64, !torch.float <-> f64,
/// !torch.vtensor <-> tensor).  Since setupBackendTypeConversion maps the
/// Torch side to the identical builtin/LLVM type, the adapted operand and
/// result types match and the op can be replaced by its operand directly.
template <typename OpType>
class ConvertDirectOp : public mlir::OpConversionPattern<OpType> {
public:
  using mlir::OpConversionPattern<OpType>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperand());
    return mlir::success();
  }
};

class ConvertTorchConversionToLLVMPass
    : public impl::ConvertTorchConversionToLLVMBase<
          ConvertTorchConversionToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter typeConverter(&context);
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);

    setupBackendTypeConversion(target, typeConverter);

    // Let tensor types pass through unchanged; a full tensor->memref->LLVM
    // pipeline is needed to lower them further.
    typeConverter.addConversion(
        [](mlir::TensorType type) -> mlir::Type { return type; });

    // Mark func.func and return as unconditionally legal — they carry
    // types that have been set up by the type converter above.
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp>();

    populateTorchConversionToLLVMConversionPatterns(target, typeConverter,
                                                    patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct TorchConversionToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateTorchConversionToLLVMConversionPatterns(target, typeConverter,
                                                    patterns);
  }
};

} // namespace

void populateTorchConversionToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  setupBackendTypeConversion(target, typeConverter);

  // Order matches TorchConversionOps.td.
  patterns
      .add<ConvertDirectOp<mlir::torch::TorchConversion::ToBuiltinTensorOp>,
           ConvertDirectOp<mlir::torch::TorchConversion::FromBuiltinTensorOp>,
           ConvertDirectOp<mlir::torch::TorchConversion::ToI1Op>,
           ConvertDirectOp<mlir::torch::TorchConversion::FromI1Op>,
           ConvertDirectOp<mlir::torch::TorchConversion::ToI64Op>,
           ConvertDirectOp<mlir::torch::TorchConversion::FromI64Op>,
           ConvertDirectOp<mlir::torch::TorchConversion::ToF64Op>,
           ConvertDirectOp<mlir::torch::TorchConversion::FromF64Op>>(
          typeConverter, patterns.getContext());

  target.addIllegalDialect<
      mlir::torch::TorchConversion::TorchConversionDialect>();
  target.addLegalDialect<mlir::BuiltinDialect, mlir::LLVM::LLVMDialect>();
}

void registerConvertTorchConversionToLLVMInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx,
          mlir::torch::TorchConversion::TorchConversionDialect *dialect) {
        dialect->addInterfaces<TorchConversionToLLVMDialectInterface>();
      });
}

} // namespace trident::torch
