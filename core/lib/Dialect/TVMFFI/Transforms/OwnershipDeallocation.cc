//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/TVMFFI/Transforms/OwnershipDeallocation.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIDialect.h" // NOLINT(misc-include-cleaner)
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <functional>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/Analysis/Liveness.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <utility>

namespace trident::tvm_ffi {

namespace {

bool isObjectCapableType(mlir::Type type) {
  return type.hasTrait<mlir::TypeTrait::Object>() ||
         mlir::isa<tvm_ffi::AnyType, tvm_ffi::UnionType>(type);
}

void materializeReferences(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::ValueRange retainedValues,
    const llvm::MapVector<mlir::Value, int64_t> &credits) {
  // Retain first because a retained value may alias a credit released below.
  for (const mlir::Value value : retainedValues) {
    tvm_ffi::ObjectIncRefOp::create(builder, loc, value);
  }
  for (auto [value, count] : credits) {
    for (int64_t reference = 0; reference < count; ++reference) {
      tvm_ffi::ObjectDecRefOp::create(builder, loc, value);
    }
  }
}

} // namespace

class EdgePlan final {
public:
  explicit EdgePlan(uint32_t successorIndex, mlir::Block *target,
                    mlir::ValueRange forwardedOperands,
                    mlir::ValueRange retainedValues,
                    const llvm::MapVector<mlir::Value, int64_t> &credits);

  void materialize(mlir::Operation *terminator) const;

private:
  uint32_t successorIndex;
  mlir::Block *target;
  llvm::SmallVector<mlir::Value> forwardedOperands;
  llvm::SmallVector<mlir::Value> retainedValues;
  std::reference_wrapper<const llvm::MapVector<mlir::Value, int64_t>> credits;
};

class BlockPlan final {
public:
  explicit BlockPlan(mlir::Block *block,
                     llvm::ArrayRef<mlir::Value> regionValues,
                     const mlir::Liveness &liveness)
      : block(block), regionValues(regionValues),
        liveness(std::cref(liveness)) {}

  mlir::LogicalResult analyze();
  void materialize() const;

private:
  mlir::Block *block;
  llvm::ArrayRef<mlir::Value> regionValues;
  std::reference_wrapper<const mlir::Liveness> liveness;
  // Credits are the owned references this block must consume or release.
  llvm::MapVector<mlir::Value, int64_t> credits;
  llvm::SmallVector<EdgePlan> edges;
};

class OwnershipDeallocator final {
public:
  static OwnershipDeallocator build(mlir::FunctionOpInterface function);
  static mlir::LogicalResult runOn(mlir::Operation *operation);

private:
  OwnershipDeallocator(mlir::Region &region, mlir::Liveness liveness,
                       mlir::ValueRange regionValues)
      : region(region), liveness(std::move(liveness)),
        regionValues(regionValues) {}

  mlir::LogicalResult run();

  mlir::Region &region;
  mlir::Liveness liveness;
  llvm::SmallVector<mlir::Value> regionValues;
};

EdgePlan::EdgePlan(uint32_t successorIndex, mlir::Block *target,
                   mlir::ValueRange forwardedOperands,
                   mlir::ValueRange retainedValues,
                   const llvm::MapVector<mlir::Value, int64_t> &credits)
    : successorIndex(successorIndex), target(target),
      forwardedOperands(forwardedOperands), retainedValues(retainedValues),
      credits(std::cref(credits)) {}

void EdgePlan::materialize(mlir::Operation *terminator) const {
  const llvm::SmallVector<mlir::Type> argumentTypes = llvm::map_to_vector(
      forwardedOperands, [](const mlir::Value operand) -> mlir::Type {
        return operand.getType();
      });
  const llvm::SmallVector<mlir::Location> argumentLocations =
      llvm::map_to_vector(forwardedOperands,
                          [](const mlir::Value operand) -> mlir::Location {
                            return operand.getLoc();
                          });
  mlir::OpBuilder builder(terminator);
  // Put edge-specific reference operations in a dedicated trampoline block.
  mlir::Block *edgeBlock =
      builder.createBlock(target, argumentTypes, argumentLocations);
  terminator->setSuccessor(edgeBlock, successorIndex);

  materializeReferences(builder, terminator->getLoc(), retainedValues,
                        credits.get());
  mlir::cf::BranchOp::create(builder, terminator->getLoc(), target,
                             edgeBlock->getArguments());
}

mlir::LogicalResult BlockPlan::analyze() {
  if (!block->isEntryBlock()) {
    // Every object entering a block establishes one local ownership credit.
    const mlir::Liveness::ValueSetT &liveIn = liveness.get().getLiveIn(block);
    for (const mlir::Value value : llvm::concat<mlir::Value>(
             llvm::make_filter_range(
                 block->getArguments(),
                 [](const mlir::BlockArgument argument) -> bool {
                   return isObjectCapableType(argument.getType());
                 }),
             llvm::make_filter_range(
                 regionValues, [&](mlir::Value value) -> bool {
                   return value.getParentBlock() != block &&
                          liveIn.contains(value) &&
                          isObjectCapableType(value.getType());
                 }))) {
      ++credits[value];
    }
  }

  mlir::Operation *terminator = block->getTerminator();
  for (mlir::Operation &operation : block->without_terminator()) {
    if (operation.getNumRegions() != 0) {
      operation.emitError("nested regions must be lowered to CFG before "
                          "ownership deallocation");
      return mlir::failure();
    }

    tvm_ffi::ObjectOwnershipOpInterface ownership =
        mlir::dyn_cast<tvm_ffi::ObjectOwnershipOpInterface>(&operation);
    // Replay each operation's ownership contract into the block ledger.
    for (auto [index, result] : llvm::make_filter_range(
             llvm::enumerate(operation.getResults()),
             [&operation](const auto &indexedResult) -> bool {
               return !mlir::isa<mlir::SelectLikeOpInterface,
                                 mlir::CastOpInterface>(operation) &&
                      isObjectCapableType(indexedResult.value().getType());
             })) {
      if (!ownership) {
        operation.emitError("object-capable result has no ownership contract");
        return mlir::failure();
      }
      if (ownership.getObjectResultOwnership(index) ==
          tvm_ffi::ObjectResultOwnership::Owned) {
        ++credits[result];
      }
    }

    for (auto [index, operand] : llvm::make_filter_range(
             llvm::enumerate(operation.getOperands()),
             [&ownership](const auto &indexedOperand) -> bool {
               return ownership &&
                      isObjectCapableType(indexedOperand.value().getType());
             })) {
      switch (ownership.getObjectOperandOwnership(index)) {
      case tvm_ffi::ObjectOperandOwnership::Borrowed:
        break;
      case tvm_ffi::ObjectOperandOwnership::Consumed:
        if (--credits[operand] < 0) {
          operation.emitError("object reference count would become negative");
          return mlir::failure();
        }
        break;
      case tvm_ffi::ObjectOperandOwnership::Retained:
        ++credits[operand];
        break;
      }
    }
  }

  if (terminator->getNumSuccessors() == 0) {
    if (!terminator->hasTrait<mlir::OpTrait::ReturnLike>() &&
        llvm::any_of(terminator->getOperandTypes(), isObjectCapableType)) {
      terminator->emitError(
          "object-bearing exit terminator must have the ReturnLike trait");
      return mlir::failure();
    }
    return mlir::success();
  }

  mlir::BranchOpInterface branch =
      mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch) {
    terminator->emitError(
        "CFG terminator with successors must implement BranchOpInterface");
    return mlir::failure();
  }

  const mlir::Liveness::ValueSetT &liveOut = liveness.get().getLiveOut(block);
  for (uint32_t index = 0; index < terminator->getNumSuccessors(); ++index) {
    const mlir::SuccessorOperands successorOperands =
        branch.getSuccessorOperands(index);
    if (successorOperands.getProducedOperandCount() != 0) {
      terminator->emitError("produced successor operands are not supported");
      return mlir::failure();
    }

    mlir::Block *target = terminator->getSuccessor(index);
    mlir::ValueRange forwardedOperands =
        successorOperands.getForwardedOperands();
    const mlir::Liveness::ValueSetT &successorLiveIn =
        liveness.get().getLiveIn(target);
    // Transfer explicit successor operands and dominating live-through values.
    const llvm::SmallVector<mlir::Value> retainedValues =
        llvm::to_vector(llvm::concat<mlir::Value>(
            llvm::make_filter_range(forwardedOperands,
                                    [](mlir::Value value) -> bool {
                                      return isObjectCapableType(
                                          value.getType());
                                    }),
            llvm::make_filter_range(
                mlir::ValueRange(regionValues), [&](mlir::Value value) -> bool {
                  return liveOut.contains(value) &&
                         successorLiveIn.contains(value) &&
                         isObjectCapableType(value.getType());
                })));
    edges.emplace_back(index, target, forwardedOperands, retainedValues,
                       credits);
  }
  return mlir::success();
}

OwnershipDeallocator
OwnershipDeallocator::build(mlir::FunctionOpInterface function) {
  mlir::Region &region = function.getFunctionBody();
  // Preserve IR order while liveness sets only provide membership queries.
  llvm::SmallVector<mlir::Value> regionValues;
  for (mlir::Block &block : region) {
    llvm::append_range(regionValues, block.getArguments());
    for (mlir::Operation &operation : block) {
      llvm::append_range(regionValues, operation.getResults());
    }
  }
  return OwnershipDeallocator(region, mlir::Liveness(function), regionValues);
}

void BlockPlan::materialize() const {
  mlir::Operation *terminator = block->getTerminator();
  if (edges.empty() && terminator->hasTrait<mlir::OpTrait::ReturnLike>()) {
    mlir::OpBuilder builder(terminator);
    const llvm::SmallVector<mlir::Value> returnedValues =
        llvm::filter_to_vector(terminator->getOperands(),
                               [](mlir::Value value) -> bool {
                                 return isObjectCapableType(value.getType());
                               });
    materializeReferences(builder, terminator->getLoc(), returnedValues,
                          credits);
  } else {
    for (const EdgePlan &edge : edges) {
      edge.materialize(terminator);
    }
  }
}

mlir::LogicalResult OwnershipDeallocator::run() {
  // Analyze the original CFG completely before materializing new edge blocks.
  llvm::SmallVector<BlockPlan> plans =
      llvm::map_to_vector(region, [&](mlir::Block &block) -> BlockPlan {
        return BlockPlan(&block, regionValues, liveness);
      });
  for (BlockPlan &plan : plans) {
    if (mlir::failed(plan.analyze())) {
      return mlir::failure();
    }
  }

  for (const BlockPlan &plan : plans) {
    plan.materialize();
  }
  return mlir::success();
}

mlir::LogicalResult OwnershipDeallocator::runOn(mlir::Operation *operation) {
  const mlir::WalkResult result = operation->walk(
      [&](mlir::FunctionOpInterface function) -> mlir::WalkResult {
        if (function.isExternal() || function.getFunctionBody().empty()) {
          return mlir::WalkResult::advance();
        }
        OwnershipDeallocator deallocator =
            OwnershipDeallocator::build(function);
        return mlir::failed(deallocator.run()) ? mlir::WalkResult::interrupt()
                                               : mlir::WalkResult::advance();
      });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

#define GEN_PASS_DEF_OWNERSHIPDEALLOCATION
#include "trident/core/Dialect/TVMFFI/Transforms/Passes.h.inc"

class OwnershipDeallocationPass final
    : public impl::OwnershipDeallocationBase<OwnershipDeallocationPass> {
public:
  void runOnOperation() final {
    if (mlir::failed(OwnershipDeallocator::runOn(getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace trident::tvm_ffi
