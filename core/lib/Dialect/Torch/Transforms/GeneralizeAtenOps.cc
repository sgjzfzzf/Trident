//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/Torch/Transforms/GeneralizeAtenOps.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h>
#include <utility>

namespace trident::torch {

#define GEN_PASS_DEF_GENERALIZEATENOPS
#include "trident/core/Dialect/Torch/Transforms/Passes.h.inc"

class GeneralizeAtenOpPattern final : public mlir::RewritePattern {
public:
  explicit GeneralizeAtenOpPattern(mlir::MLIRContext *context)
      : RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), /*benefit=*/1,
                       context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const final {
    if (const llvm::StringRef name = op->getName().getStringRef();
        name.starts_with("torch.aten.")) {
      mlir::NamedAttrList attributes(op->getAttrs());
      attributes.set("name", rewriter.getStringAttr(name));

      if (op->getNumRegions() != 0) {
        return rewriter.notifyMatchFailure(
            op, "expected a regionless ATen operation");
      }

      if (name == "torch.aten.Int.bool") {
        if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
          return rewriter.notifyMatchFailure(
              op, "expected one operand and one result");
        }
        mlir::Value const operand = op->getOperand(0);
        rewriter.setInsertionPoint(op);
        trident::torchext::GetOp nativeBoolOp =
            trident::torchext::GetOp::create(rewriter, op->getLoc(),
                                             rewriter.getI1Type(), operand);
        mlir::Value const nativeInt = mlir::arith::ExtUIOp::create(
            rewriter, op->getLoc(), rewriter.getI64Type(),
            nativeBoolOp.getResult());
        rewriter.replaceOpWithNewOp<mlir::torch::TorchConversion::FromI64Op>(
            op, nativeInt);
        return mlir::success();
      }

      rewriter.replaceOpWithNewOp<mlir::torch::Torch::OperatorOp>(
          op, op->getResultTypes(), op->getOperands(), attributes.getAttrs(),
          /*numRegions=*/0);
      return mlir::success();
    } else {
      return mlir::failure();
    }
  }
};

class GeneralizeAtenOpsPass final
    : public impl::GeneralizeAtenOpsBase<GeneralizeAtenOpsPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<GeneralizeAtenOpPattern>(&getContext());
    mlir::walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace trident::torch
