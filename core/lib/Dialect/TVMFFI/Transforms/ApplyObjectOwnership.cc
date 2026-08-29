//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/Transforms/ApplyObjectOwnership.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

namespace trident::tvm_ffi {
static bool isObjectCapableType(mlir::Type type) {
  return type.hasTrait<mlir::TypeTrait::Object>() ||
         mlir::isa<tvm_ffi::AnyType, tvm_ffi::UnionType>(type);
}

static mlir::LogicalResult applyObjectOwnership(mlir::Operation *operation) {
  mlir::WalkResult const result = operation->walk([&](mlir::Region *region) {
    // Analyze only regions with ref-count operations or object-valued returns.
    if (region->empty() ||
        !llvm::any_of(region->getOps(), [](mlir::Operation &operation) {
          return mlir::isa<tvm_ffi::ObjectIncRefOp, tvm_ffi::ObjectDecRefOp>(
                     operation) ||
                 (operation.hasTrait<mlir::OpTrait::IsTerminator>() &&
                  llvm::any_of(operation.getOperandTypes(),
                               isObjectCapableType)) ||
                 llvm::any_of(operation.getResultTypes(), isObjectCapableType);
        })) {
      return mlir::WalkResult::advance();
    }
    if (!region->hasOneBlock()) {
      region->getParentOp()->emitError(
          "object ownership analysis requires single-block regions");
      return mlir::WalkResult::interrupt();
    }
    if (mlir::isa<mlir::LoopLikeOpInterface>(region->getParentOp())) {
      region->getParentOp()->emitError(
          "object ownership analysis does not support loop regions");
      return mlir::WalkResult::interrupt();
    }
    for (mlir::Block &block : region->getBlocks()) {
      llvm::DenseMap<mlir::Value, int64_t> counters;
      mlir::Operation *terminator = block.getTerminator();
      for (mlir::Operation &operation : block.without_terminator()) {
        for (auto [index, result] : llvm::make_filter_range(
                 llvm::enumerate(operation.getResults()),
                 [](const auto &it) -> bool {
                   auto [_, operand] = it;
                   return isObjectCapableType(operand.getType());
                 })) {
          if (mlir::isa<mlir::SelectLikeOpInterface, mlir::CastOpInterface>(
                  operation)) {
            continue;
          }
          if (auto ownership =
                  mlir::dyn_cast<tvm_ffi::ObjectOwnershipOpInterface>(
                      &operation)) {
            switch (ownership.getObjectResultOwnership(
                static_cast<uint32_t>(index))) {
            case tvm_ffi::ObjectResultOwnership::Borrowed:
              break;
            case tvm_ffi::ObjectResultOwnership::Owned:
              ++counters[result];
              break;
            }
          } else {
            operation.emitError(
                "object-capable result has no ownership contract");
            return mlir::WalkResult::interrupt();
          }
        }
        if (auto ownership =
                mlir::dyn_cast<tvm_ffi::ObjectOwnershipOpInterface>(
                    &operation)) {
          for (auto [index, operand] : llvm::make_filter_range(
                   llvm::enumerate(operation.getOperands()),
                   [](const auto &it) -> bool {
                     auto [_, operand] = it;
                     return isObjectCapableType(operand.getType());
                   })) {
            switch (ownership.getObjectOperandOwnership(
                static_cast<uint32_t>(index))) {
            case tvm_ffi::ObjectOperandOwnership::Borrowed:
              break;
            case tvm_ffi::ObjectOperandOwnership::Consumed:
              if (--counters[operand] < 0) {
                operation.emitError(
                    "object reference count would become negative");
                return mlir::WalkResult::interrupt();
              }
              break;
            case tvm_ffi::ObjectOperandOwnership::Retained:
              ++counters[operand];
              break;
            }
          }
        }
      }

      mlir::OpBuilder builder(terminator);
      mlir::Location const loc = terminator->getLoc();
      for (mlir::Value const operand : llvm::make_filter_range(
               terminator->getOperands(), [](mlir::Value operand) -> bool {
                 return isObjectCapableType(operand.getType());
               })) {
        tvm_ffi::ObjectIncRefOp::create(builder, loc, operand);
      }
      for (auto [value, count] : counters) {
        for (int64_t reference = 0; reference < count; ++reference) {
          tvm_ffi::ObjectDecRefOp::create(builder, loc, value);
        }
      }
    }
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

#define GEN_PASS_DEF_APPLYOBJECTOWNERSHIP
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

class ApplyObjectOwnershipPass final
    : public impl::ApplyObjectOwnershipBase<ApplyObjectOwnershipPass> {
public:
  void runOnOperation() final {
    if (mlir::failed(applyObjectOwnership(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::tvm_ffi
