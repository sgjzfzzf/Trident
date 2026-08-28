//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/ArithExt/IR/ArithExtDialect.h"

#include <cstdint>

#include "trident/core/Dialect/ArithExt/IR/ArithExtDialect.cpp.inc"
#include "trident/core/Dialect/ArithExt/IR/ArithExtOps.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Support/LLVM.h>
#define GET_OP_CLASSES
#include "trident/core/Dialect/ArithExt/IR/ArithExt.cpp.inc"

namespace trident::arithext {

void ArithExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "trident/core/Dialect/ArithExt/IR/ArithExt.cpp.inc"
      >();
}

void AndThenOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &successors) {
  if (getNumRegions() == 0) {
  } else if (point.isParent()) {
    successors.emplace_back(&getRegion(0));
  } else {
    const mlir::RegionBranchTerminatorOpInterface terminator =
        point.getTerminatorPredecessorOrNull();
    getSuccessorRegions(*terminator->getParentRegion(), successors);
  }
}

void AndThenOp::getSuccessorRegions(
    mlir::Region &region,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &successors) {
  const uint32_t regionIndex = region.getRegionNumber();
  successors.emplace_back(regionIndex + 1 < getNumRegions()
                              ? &getRegion(regionIndex + 1)
                              : mlir::RegionSuccessor::parent());
}

mlir::ValueRange
AndThenOp::getSuccessorInputs(mlir::RegionSuccessor successor) {
  return successor.isParent() ? getOperation()->getResults()
                              : mlir::ValueRange{};
}

mlir::MutableOperandRange
AndThenYieldOp::getMutableSuccessorOperands(mlir::RegionSuccessor successor) {
  return mlir::MutableOperandRange(getOperation(), 0, successor.isParent());
}

void AndThenYieldOp::getSuccessorRegions(
    llvm::ArrayRef<mlir::Attribute>,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &successors) {
  mlir::cast<AndThenOp>(getOperation()->getParentOp())
      .getSuccessorRegions(mlir::RegionBranchPoint(*this), successors);
}

} // namespace trident::arithext
