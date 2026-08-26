//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_ARITHEXT_IR_ARITHEXT_OPS_TD_
#define TRIDENT_CORE_DIALECT_ARITHEXT_IR_ARITHEXT_OPS_TD_

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "trident/core/Dialect/ArithExt/IR/ArithExtBase.td"

class ArithExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<ArithExt_Dialect, mnemonic, traits>;

def ArithExt_AndThenOp : ArithExt_Op<"and_then", [RegionBranchOpInterface]> {
  let summary = "Execute regions sequentially until one yields false";
  let description = [{
    Executes each region in order. A region yields `true` to continue with the
    next region and yields `false` to stop execution and return `false` from
    this operation. If all regions yield `true`, this operation returns
    `true`. An operation with no regions returns `true`.
  }];

  let results = (outs I1:$result);
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void getSuccessorRegions(
        ::mlir::RegionBranchPoint point,
        ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
    void getSuccessorRegions(
        ::mlir::Region &region,
        ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
    ::mlir::ValueRange getSuccessorInputs(::mlir::RegionSuccessor successor);
  }];

}

def ArithExt_AndThenYieldOp : ArithExt_Op<"and_then.yield", [
  HasParent<"AndThenOp">,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
    ["getMutableSuccessorOperands", "getSuccessorRegions"]>,
  RegionBranchTerminatorOpInterface,
  Terminator
]> {
  let summary = "Yield a boolean result from an and_then region";
  let arguments = (ins I1:$value);

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // TRIDENT_CORE_DIALECT_ARITHEXT_IR_ARITHEXT_OPS_TD_
