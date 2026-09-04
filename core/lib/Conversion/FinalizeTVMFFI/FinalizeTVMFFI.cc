//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Conversion/FinalizeTVMFFI/FinalizeTVMFFI.h"
#include "trident/core/Dialect/DLPack/IR/DLPackDialect.h"
#include "trident/core/Dialect/DLPack/IR/DLPackOps.h"
#include "trident/core/Dialect/DLPack/IR/DLPackTypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace trident::conversion {

#define GEN_PASS_DEF_FINALIZETVMFFI
#include "trident/core/Conversion/Passes.h.inc"

template <typename SourceOp, typename TargetOp>
class FinalizeIndexedTensorMetadataOp final
    : public mlir::OpRewritePattern<SourceOp> {
public:
  using mlir::OpRewritePattern<SourceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Type const objectType = tvm_ffi::ObjectType::get(context);
    mlir::Type const tensorType = dlpack::DLTensorType::get(context);
    mlir::Value const object = tvm_ffi::GetOp::create(
        rewriter, op.getLoc(), objectType, op.getTensor());
    mlir::Value const tensor =
        tvm_ffi::AsOp::create(rewriter, op.getLoc(), tensorType, object);
    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getResult().getType(), tensor,
                                          op.getIndex());
    return mlir::success();
  }
};

template <typename SourceOp, typename TargetOp>
class FinalizeTensorMetadataOp final : public mlir::OpRewritePattern<SourceOp> {
public:
  using mlir::OpRewritePattern<SourceOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = rewriter.getContext();
    mlir::Type const objectType = tvm_ffi::ObjectType::get(context);
    mlir::Type const tensorType = dlpack::DLTensorType::get(context);
    mlir::Value const object = tvm_ffi::GetOp::create(
        rewriter, op.getLoc(), objectType, op.getTensor());
    mlir::Value const tensor =
        tvm_ffi::AsOp::create(rewriter, op.getLoc(), tensorType, object);
    rewriter.replaceOpWithNewOp<TargetOp>(op, op->getResultTypes(), tensor);
    return mlir::success();
  }
};

void populateFinalizeTVMFFIPatterns(mlir::RewritePatternSet &patterns) {
  patterns.add<
      FinalizeTensorMetadataOp<tvm_ffi::TensorDeviceOp, dlpack::TensorDeviceOp>,
      FinalizeTensorMetadataOp<tvm_ffi::TensorDimOp, dlpack::TensorDimOp>,
      FinalizeTensorMetadataOp<tvm_ffi::TensorDTypeOp, dlpack::TensorDTypeOp>,
      FinalizeIndexedTensorMetadataOp<tvm_ffi::TensorSizeOp,
                                      dlpack::TensorSizeOp>,
      FinalizeTensorMetadataOp<tvm_ffi::TensorStorageOffsetOp,
                               dlpack::TensorStorageOffsetOp>,
      FinalizeIndexedTensorMetadataOp<tvm_ffi::TensorStrideOp,
                                      dlpack::TensorStrideOp>>(
      patterns.getContext());
}

class FinalizeTVMFFIPass final
    : public impl::FinalizeTVMFFIBase<FinalizeTVMFFIPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    populateFinalizeTVMFFIPatterns(patterns);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::conversion
