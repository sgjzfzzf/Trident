#include "libtriton-core/Dialect/TorchExt/Transforms/RewriteTorchAsTorchExt.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "libtriton-core/Dialect/TorchExt/IR/TorchExtOps.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

namespace libtriton::torchext {

#define GEN_PASS_DEF_REWRITETORCHASTORCHEXT
#include "libtriton-core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

/// Rewrites any Torch dialect op whose name starts with "aten." into a
/// TorchExt torchext.call_dispatcher op. The "aten." prefix is replaced
/// with "aten::" as the dispatcher op_name, and the overload_name is empty.
struct ConvertAtenOp : public mlir::RewritePattern {
  ConvertAtenOp(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(),
                             /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<mlir::torch::Torch::TorchDialect>(op->getDialect())) {
      return mlir::failure();
    }

    llvm::StringRef opName = op->getName().getStringRef();
    constexpr llvm::StringRef kAtenPrefix = "torch.aten.";
    if (!opName.starts_with(kAtenPrefix)) {
      return mlir::failure();
    }

    // Build op_name: strip "torch." prefix, then replace "aten." with "aten::"
    llvm::Twine newOpName = "aten::" + opName.drop_front(kAtenPrefix.size());

    rewriter.replaceOpWithNewOp<libtriton::torchext::CallDispatcherOp>(
        op, op->getResultTypes(), rewriter.getStringAttr(newOpName),
        rewriter.getStringAttr(""), op->getOperands());
    return mlir::success();
  }
};

class RewriteTorchAsTorchExtPass
    : public libtriton::torchext::impl::RewriteTorchAsTorchExtBase<
          RewriteTorchAsTorchExtPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::RewritePatternSet patterns(&context);
    populateRewriteTorchAsTorchExtPatterns(patterns);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateRewriteTorchAsTorchExtPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertAtenOp>(patterns.getContext());
}

} // namespace libtriton::torchext
