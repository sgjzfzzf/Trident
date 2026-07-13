//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_
#define TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def RAAI : Pass<"raai", "mlir::ModuleOp"> {
  let summary =
      "Release allocated intermediate variables in single-block regions";
  let description = [{
    Walks every op in the module.  For each single-block region (func::FuncOp,
    scf.if / scf.for bodies, etc.) it inserts torchext.ObjectIncRef for every
    yielded value and torchext.ObjectDecRef for every internally-allocated
    value (torch.prim.ListConstruct and torch.aten.* tensor/list results).
  }];
  let dependentDialects = ["trident::torchext::TorchExtDialect"];
}

def EliminateRefCountPairs
    : Pass<"eliminate-ref-count-pairs", "mlir::ModuleOp"> {
  let summary = "Eliminate balanced reference-count operations in a block";
  let description = [{
    Walks every block in the module and pairs each torchext.ObjectDecRef with
    the nearest preceding unmatched torchext.ObjectIncRef on the same SSA
    value. Both operations in every pair are erased. Matching never crosses
    a block boundary, and a DecRef that precedes an IncRef is not eliminated.
  }];
  let dependentDialects = ["trident::torchext::TorchExtDialect"];
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_TRANSFORMS_PASSES_TD_
