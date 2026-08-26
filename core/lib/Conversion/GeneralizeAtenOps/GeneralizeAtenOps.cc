//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/GeneralizeAtenOps/GeneralizeAtenOps.h" // NOLINT(misc-include-cleaner)
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>
#include <torch-mlir/Dialect/Torch/IR/TorchOps.h>
#include <utility>

namespace trident::torch {

#define GEN_PASS_DEF_GENERALIZEATENOPS
#include "trident/core/Conversion/Passes.h.inc"

namespace {

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

} // namespace

} // namespace trident::torch
