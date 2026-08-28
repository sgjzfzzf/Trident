//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops --mlir-print-debuginfo | FileCheck %s --check-prefix=GENERALIZE
// RUN: trident-core-opt %s -generalize-aten-ops --mlir-print-debuginfo | FileCheck %s --check-prefix=LOCATION
// RUN: trident-core-opt %s -generalize-aten-ops -generalize-aten-ops | FileCheck %s --check-prefix=IDEMPOTENT
// RUN: trident-core-opt %s -generalize-aten-ops -canonicalize | FileCheck %s --check-prefix=CANONICALIZE

// GENERALIZE-LABEL: func.func @single_result
// GENERALIZE: %[[FORMAT:[a-zA-Z0-9_]+]] = torch.constant.int 0
// GENERALIZE: %[[CLONE:[a-zA-Z0-9_]+]] = torch.operator "torch.aten.clone"(%arg0, %[[FORMAT]]) {trident.test = "kept"}
// GENERALIZE-SAME: -> !torch.vtensor<[3,2],f32>
// GENERALIZE: return %[[CLONE]]
// LOCATION: %[[CLONE_RESULT:[a-zA-Z0-9_]+]] = torch.operator "torch.aten.clone"(%arg0, %[[CLONE_FORMAT:[a-zA-Z0-9_]+]]) {trident.test = "kept"} : (!torch.vtensor<[3,2],f32>, !torch.int) -> !torch.vtensor<[3,2],f32> loc(#[[CLONE_LOC:loc[0-9]+]])
// LOCATION: #[[CLONE_LOC]] = loc("model.py":42:7)
// IDEMPOTENT-LABEL: func.func @single_result
// IDEMPOTENT-COUNT-1: torch.operator "torch.aten.clone"
// CANONICALIZE-LABEL: func.func @single_result
// CANONICALIZE: %[[CLONE:[a-zA-Z0-9_]+]] = torch.operator "torch.aten.clone"(%arg0,
// CANONICALIZE: return %[[CLONE]]
func.func @single_result(%arg0: !torch.vtensor<[3,2],f32>)
    -> !torch.vtensor<[3,2],f32> {
  %format = torch.constant.int 0
  %clone = "torch.aten.clone"(%arg0, %format) {trident.test = "kept"}
      : (!torch.vtensor<[3,2],f32>, !torch.int)
      -> !torch.vtensor<[3,2],f32> loc("model.py":42:7)
  return %clone : !torch.vtensor<[3,2],f32>
}

// CANONICALIZE-LABEL: func.func @other_dialect_folding
// CANONICALIZE: %[[THREE:[a-zA-Z0-9_]+]] = arith.constant 3 : i64
// CANONICALIZE-NEXT: return %[[THREE]] : i64
func.func @other_dialect_folding() -> i64 {
  %one = arith.constant 1 : i64
  %two = arith.constant 2 : i64
  %sum = arith.addi %one, %two : i64
  return %sum : i64
}

// GENERALIZE-LABEL: func.func @multiple_results
// GENERALIZE: %[[PAIR:[a-zA-Z0-9_]+]]:2 = torch.operator "torch.aten.max.dim"
// GENERALIZE: return %[[PAIR]]#0, %[[PAIR]]#1
func.func @multiple_results(%arg0: !torch.vtensor<[4],f32>)
    -> (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) {
  %dim = torch.constant.int 0
  %keepdim = torch.constant.bool false
  %values, %indices = torch.aten.max.dim %arg0, %dim, %keepdim
      : !torch.vtensor<[4],f32>, !torch.int, !torch.bool
      -> !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
  return %values, %indices
      : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>
}

// GENERALIZE-LABEL: func.func @zero_results
// GENERALIZE: torch.operator "torch.aten._assert_tensor_metadata"
func.func @zero_results(%arg0: !torch.vtensor<[4],f32>) {
  %none = torch.constant.none
  torch.aten._assert_tensor_metadata %arg0, %none, %none, %none, %none, %none
      : !torch.vtensor<[4],f32>, !torch.none, !torch.none, !torch.none,
        !torch.none, !torch.none
  return
}

// GENERALIZE-LABEL: func.func @preserve_structural_ops
// GENERALIZE: %[[INT:[a-zA-Z0-9_]+]] = torch.constant.int 0
// GENERALIZE: %[[LIST:[a-zA-Z0-9_]+]] = torch.prim.ListConstruct %[[INT]]
// GENERALIZE: torch.runtime.assert %arg0, "condition"
// GENERALIZE: %[[OPAQUE:[a-zA-Z0-9_]+]] = torch.operator "torch.aten.existing"
// GENERALIZE: return %[[OPAQUE]]
func.func @preserve_structural_ops(%arg0: !torch.bool)
    -> !torch.vtensor<[1],f32> {
  %int = torch.constant.int 0
  %list = torch.prim.ListConstruct %int : (!torch.int) -> !torch.list<int>
  torch.runtime.assert %arg0, "condition"
  %opaque = torch.operator "torch.aten.existing"(%list)
      {trident.test = "unchanged"}
      : (!torch.list<int>) -> !torch.vtensor<[1],f32>
  return %opaque : !torch.vtensor<[1],f32>
}

// GENERALIZE-LABEL: func.func @nested_region
// GENERALIZE: scf.if
// GENERALIZE: %[[TRANSPOSED:[a-zA-Z0-9_]+]] = torch.operator "torch.aten.t"(%arg0)
// GENERALIZE: scf.yield %[[TRANSPOSED]]
func.func @nested_region(%arg0: !torch.vtensor<[2,3],f32>, %cond: i1)
    -> !torch.vtensor<[3,2],f32> {
  %result = scf.if %cond -> (!torch.vtensor<[3,2],f32>) {
    %transposed = torch.aten.t %arg0
        : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    scf.yield %transposed : !torch.vtensor<[3,2],f32>
  } else {
    %transposed = torch.aten.t %arg0
        : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    scf.yield %transposed : !torch.vtensor<[3,2],f32>
  }
  return %result : !torch.vtensor<[3,2],f32>
}
