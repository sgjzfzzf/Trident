//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/Torch/Transforms/GeneralizeAtenOps.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <torch-mlir/Dialect/Torch/IR/TorchDialect.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h>
#include <utility>

namespace trident::torch {

#define GEN_PASS_DEF_GENERALIZEATENOPS
#include "GeneralizeAtenOpsPDLLPatterns.h.inc"
#include "trident/core/Dialect/Torch/Transforms/Passes.h.inc"

class GeneralizeAtenOpPattern final : public mlir::RewritePattern {
public:
  explicit GeneralizeAtenOpPattern(mlir::MLIRContext *context)
      : RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), /*benefit=*/1,
                       context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const final {
    const llvm::StringRef name = op->getName().getStringRef();
    if (!name.starts_with("torch.aten.")) {
      return mlir::failure();
    }
    mlir::NamedAttrList attributes(op->getAttrs());
    attributes.set("name", rewriter.getStringAttr(name));

    if (op->getNumRegions() != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected a regionless ATen operation");
    }
    rewriter.replaceOpWithNewOp<mlir::torch::Torch::OperatorOp>(
        op, op->getResultTypes(), op->getOperands(), attributes.getAttrs(),
        /*numRegions=*/0);
    return mlir::success();
  }
};

class GeneralizeAtenOpsPass final
    : public impl::GeneralizeAtenOpsBase<GeneralizeAtenOpsPass> {
public:
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addIllegalOp<
        mlir::torch::Torch::AtenAddIntOp, mlir::torch::Torch::AtenFloordivIntOp,
        mlir::torch::Torch::AtenIntBoolOp, mlir::torch::Torch::AtenSizeIntOp,
        mlir::torch::Torch::AtenSubIntOp>();
    target.addLegalDialect<
        mlir::BuiltinDialect, mlir::func::FuncDialect,
        mlir::arith::ArithDialect, mlir::torch::Torch::TorchDialect,
        mlir::torch::TorchConversion::TorchConversionDialect,
        trident::tvm_ffi::TVMFFIDialect, trident::torchext::TorchExtDialect>();
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) -> bool { return true; });

    mlir::RewritePatternSet specializedPatterns(&getContext());
    populateGeneratedPDLLPatterns(specializedPatterns);
    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(specializedPatterns)))) {
      signalPassFailure();
      return;
    }

    mlir::RewritePatternSet generalPatterns(&getContext());
    generalPatterns.add<GeneralizeAtenOpPattern>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(generalPatterns));
  }
};

} // namespace trident::torch
