//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident/core/Dialect/ArithExt/IR/ArithExtOps.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>

namespace trident::arithext {

mlir::LogicalResult AndThenOp::verify() {
  for (mlir::Region &region : getRegions()) {
    if (region.empty() || !llvm::hasSingleElement(region)) {
      return emitOpError("expects each region to contain exactly one block");
    } else if (region.front().empty() ||
               !mlir::isa<AndThenYieldOp>(region.front().back())) {
      return emitOpError(
          "expects each region to terminate with arithext.and_then.yield");
    }
  }
  return mlir::success();
}

} // namespace trident::arithext
