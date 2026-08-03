//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/TorchToLLVM/TorchToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "trident/core/Dialect/TorchExt/Transforms/BackendTypeConversion.h"

namespace trident::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOLLVM
#include "trident/core/Conversion/Passes.h.inc"

namespace {

class ConvertTorchToLLVMPass
    : public impl::ConvertTorchToLLVMBase<ConvertTorchToLLVMPass> {
public:
  void runOnOperation() final {
    mlir::LLVMTypeConverter typeConverter(&getContext());
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    populateTorchToLLVMConversionPatterns(target, typeConverter, patterns);

    // Fold only as a fallback *after* running the conversion patterns.
    // With the default `BeforePatterns` mode, torch.aten.clone (whose fold
    // unconditionally returns the self operand when the types match, ignoring
    // memory_format) gets folded before ConvertAtenDispatcherOp can lower it to
    // the trident.aten.clone FFI call. That would alias the clone result with
    // its operand, unbalancing the reference counting inserted by RAAI and
    // causing a premature release / double free at runtime.
    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(patterns),
            mlir::ConversionConfig{
                .foldingMode =
                    mlir::DialectConversionFoldingMode::AfterPatterns}))) {
      signalPassFailure();
    }
  }
};

struct TorchToLLVMDialectInterface
    : public mlir::ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    populateTorchToLLVMConversionPatterns(target, typeConverter, patterns);
  }
};

} // namespace

void populateTorchToLLVMConversionPatterns(
    mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns) {
  setupBackendTypeConversion(target, typeConverter);
  populateTorchToLLVMConstantConversionPatterns(target, typeConverter,
                                                patterns);
  populateTorchToLLVMLiteralConversionPatterns(target, typeConverter, patterns);
  populateTorchToLLVMAtenConversionPatterns(target, typeConverter, patterns);
  populateTorchToLLVMPrimConversionPatterns(target, typeConverter, patterns);
  target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::BuiltinDialect,
                         mlir::func::FuncDialect>();
}

void registerConvertTorchToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, mlir::torch::Torch::TorchDialect *dialect) {
        dialect->addInterfaces<TorchToLLVMDialectInterface>();
      });
}

} // namespace trident::torch
