#include "libtriton-core/Conversion/TorchToAOTInductor/TorchToAOTInductor.h"
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorDialect.h"
#include "libtriton-core/Dialect/AOTInductor/IR/AOTInductorOps.h"

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

namespace libtriton::torch {

#define GEN_PASS_DEF_CONVERTTORCHTOAOTINDUCTOR
#include "libtriton-core/Conversion/Passes.h.inc"

namespace {

/// Converts any Torch dialect op whose name starts with "aten." into an
/// AOTInductor aoti.torch_call_dispatcher op. The "aten." prefix is replaced
/// with "aten::" as the dispatcher op_name, and the overload_name is empty.
struct ConvertAtenOp : public mlir::ConversionPattern {
  /// Uses the inherited RewritePattern(MatchAnyOpTypeTag, ...) constructor,
  /// leaving TypeConverter as nullptr since no type conversion is needed.
  ConvertAtenOp(mlir::MLIRContext *context)
      : mlir::ConversionPattern(mlir::Pattern::MatchAnyOpTypeTag(),
                                /*benefit=*/1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<mlir::torch::Torch::TorchDialect>(op->getDialect())) {
      return mlir::failure();
    }

    llvm::StringRef opName = op->getName().getStringRef();
    constexpr llvm::StringRef kAtenPrefix = "torch.aten.";
    if (!opName.starts_with(kAtenPrefix)) {
      return mlir::failure();
    }

    // Build op_name: strip "torch." prefix, then replace "aten." with "aten::"
    std::string newOpName =
        ("aten::" + opName.drop_front(kAtenPrefix.size())).str();

    rewriter.replaceOpWithNewOp<libtriton::aoti::TorchCallDispatcherOp>(
        op, op->getResultTypes(), rewriter.getStringAttr(newOpName),
        rewriter.getStringAttr(""), operands);
    return mlir::success();
  }
};

class ConvertTorchToAOTInductorPass
    : public impl::ConvertTorchToAOTInductorBase<
          ConvertTorchToAOTInductorPass> {
public:
  void runOnOperation() final {
    mlir::MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    populateTorchToAOTInductorConversionPatterns(target, patterns);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateTorchToAOTInductorConversionPatterns(
    mlir::ConversionTarget &target, mlir::RewritePatternSet &patterns) {
  patterns.add<ConvertAtenOp>(patterns.getContext());
  // Only aten.* ops are illegal (triggering ConvertAtenOp); all other Torch ops
  // (e.g. torch.constant.none) and non-Torch ops remain legal.
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addDynamicallyLegalDialect<mlir::torch::Torch::TorchDialect>(
      [](mlir::Operation *op) {
        return !op->getName().getStringRef().starts_with("torch.aten.");
      });
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
}

} // namespace libtriton::torch
