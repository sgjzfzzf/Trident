//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtDialect.h"
#include "trident/core/Dialect/TorchExt/IR/TorchExtOps.h"

namespace trident::torch {

#define GEN_PASS_DEF_RAAI
#define GEN_PASS_REGISTRATION_RAAI
#include "trident/core/Dialect/TorchExt/Transforms/Passes.h.inc"

namespace {

class RAAIPass : public impl::RAAIBase<RAAIPass> {
public:
  void runOnOperation() final {
    getOperation().walk([&](mlir::Operation *op) {
      // Skip the module itself — its body is not a meaningful scope
      // for per-block reference counting.
      if (llvm::isa<mlir::ModuleOp>(op)) {
        return;
      }
      for (mlir::Region &region : op->getRegions()) {
        if (!region.hasOneBlock()) {
          continue;
        }
        mlir::Block &block = region.front();

        // --- Set insertion point before the terminator ---
        mlir::Operation *terminator = block.getTerminator();
        mlir::OpBuilder builder(terminator);
        mlir::Location loc = terminator->getLoc();

        // IncRef all yielded values (they escape this scope).
        for (mlir::Value val : terminator->getOperands()) {
          torchext::ObjectIncRefOp::create(builder, loc, val);
        }

        // Walk the block and insert DecRef for each heap-allocated result.
        block.walk([&](mlir::Operation *op) {
          // Skip pure cast ops (e.g. torch.derefine) — they do not create a
          // new heap allocation; the underlying value is owned by its producer.
          if (!llvm::isa<mlir::CastOpInterface>(op)) {
            for (mlir::Value result : op->getResults()) {
              mlir::Type ty = result.getType();
              // TODO: also handle DictType, NnModuleType, GeneratorType,
              //       LinearParamsType when they appear in practice.
              if (llvm::isa<mlir::torch::Torch::ValueTensorType,
                            mlir::torch::Torch::NonValueTensorType,
                            mlir::torch::Torch::ListType,
                            mlir::torch::Torch::TupleType>(ty)) {
                torchext::ObjectDecRefOp::create(builder, loc, result);
              } else if (llvm::isa<mlir::torch::Torch::OptionalType>(ty)) {
                mlir::OpBuilder::InsertionGuard guard(builder);
                // Runtime None check: DecRef only if not None.
                mlir::torch::Torch::ConstantNoneOp noneOp =
                    mlir::torch::Torch::ConstantNoneOp::create(builder, loc);
                mlir::torch::Torch::Aten__Isnot__Op isNotNoneOp =
                    mlir::torch::Torch::Aten__Isnot__Op::create(
                        builder, loc, result, noneOp.getResult());

                mlir::torch::Torch::PrimIfOp ifOp =
                    mlir::torch::Torch::PrimIfOp::create(
                        builder, loc,
                        /*resultTypes=*/{}, isNotNoneOp.getResult());

                // thenRegion: value is not None → DecRef
                mlir::Block &thenBlock = ifOp.getThenRegion().emplaceBlock();
                builder.setInsertionPointToStart(&thenBlock);
                torchext::ObjectDecRefOp::create(builder, loc, result);
                mlir::torch::Torch::PrimIfYieldOp::create(builder, loc,
                                                          /*results=*/{});

                // elseRegion: value is None → no-op
                mlir::Block &elseBlock = ifOp.getElseRegion().emplaceBlock();
                builder.setInsertionPointToStart(&elseBlock);
                mlir::torch::Torch::PrimIfYieldOp::create(builder, loc,
                                                          /*results=*/{});
              }
            }
          }
        });
      }
    });
  }
};

} // namespace

} // namespace trident::torch
