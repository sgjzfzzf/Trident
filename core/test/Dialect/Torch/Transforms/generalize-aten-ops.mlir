//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops --mlir-print-debuginfo | FileCheck %s --check-prefix=GENERALIZE
// RUN: trident-core-opt %s -generalize-aten-ops --mlir-print-debuginfo | FileCheck %s --check-prefix=LOCATION
// RUN: trident-core-opt %s -generalize-aten-ops -generalize-aten-ops | FileCheck %s --check-prefix=IDEMPOTENT

// GENERALIZE-LABEL: func.func @single_result
// GENERALIZE: %[[FORMAT:[a-zA-Z0-9_]+]] = torch.constant.int 0
// GENERALIZE: %[[CLONE:[a-zA-Z0-9_]+]] = torch.operator "torch.aten.clone"(%arg0, %[[FORMAT]]) {trident.test = "kept"}
// GENERALIZE-SAME: -> !torch.vtensor<[3,2],f32>
// GENERALIZE: return %[[CLONE]]
// LOCATION: %[[CLONE_RESULT:[a-zA-Z0-9_]+]] = torch.operator "torch.aten.clone"(%arg0, %[[CLONE_FORMAT:[a-zA-Z0-9_]+]]) {trident.test = "kept"} : (!torch.vtensor<[3,2],f32>, !torch.int) -> !torch.vtensor<[3,2],f32> loc(#[[CLONE_LOC:loc[0-9]+]])
// LOCATION: #[[CLONE_LOC]] = loc("model.py":42:7)
// IDEMPOTENT-LABEL: func.func @single_result
// IDEMPOTENT-COUNT-1: torch.operator "torch.aten.clone"
func.func @single_result(%arg0: !torch.vtensor<[3,2],f32>)
    -> !torch.vtensor<[3,2],f32> {
  %format = torch.constant.int 0
  %clone = "torch.aten.clone"(%arg0, %format) {trident.test = "kept"}
      : (!torch.vtensor<[3,2],f32>, !torch.int)
      -> !torch.vtensor<[3,2],f32> loc("model.py":42:7)
  return %clone : !torch.vtensor<[3,2],f32>
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

// GENERALIZE-LABEL: func.func @specialized_add_int
// GENERALIZE: %[[ADD_A:[a-zA-Z0-9_]+]] = torchext.get %arg0 : !torch.int -> i64
// GENERALIZE: %[[ADD_B:[a-zA-Z0-9_]+]] = torchext.get %arg1 : !torch.int -> i64
// GENERALIZE: %[[ADD_RESULT:[a-zA-Z0-9_]+]] = arith.addi %[[ADD_A]], %[[ADD_B]] : i64
// GENERALIZE: %[[ADD_INT:[a-zA-Z0-9_]+]] = torch_c.from_i64 %[[ADD_RESULT]]
func.func @specialized_add_int(%arg0: !torch.int, %arg1: !torch.int)
    -> !torch.int {
  %result = torch.aten.add.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}

// GENERALIZE-LABEL: func.func @specialized_floordiv_int
// GENERALIZE: %[[FLOOR_A:[a-zA-Z0-9_]+]] = torchext.get %arg0 : !torch.int -> i64
// GENERALIZE: %[[FLOOR_B:[a-zA-Z0-9_]+]] = torchext.get %arg1 : !torch.int -> i64
// GENERALIZE: %[[FLOOR_RESULT:[a-zA-Z0-9_]+]] = arith.floordivsi %[[FLOOR_A]], %[[FLOOR_B]] : i64
// GENERALIZE: torch_c.from_i64 %[[FLOOR_RESULT]]
func.func @specialized_floordiv_int(%arg0: !torch.int, %arg1: !torch.int)
    -> !torch.int {
  %result = torch.aten.floordiv.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}

// GENERALIZE-LABEL: func.func @specialized_int_bool
// GENERALIZE: %[[BOOL_NATIVE:[a-zA-Z0-9_]+]] = torchext.get %arg0 : !torch.bool -> i1
// GENERALIZE: %[[BOOL_INT:[a-zA-Z0-9_]+]] = arith.extui %[[BOOL_NATIVE]] : i1 to i64
// GENERALIZE: torch_c.from_i64 %[[BOOL_INT]]
func.func @specialized_int_bool(%arg0: !torch.bool) -> !torch.int {
  %result = torch.aten.Int.bool %arg0 : !torch.bool -> !torch.int
  return %result : !torch.int
}

// GENERALIZE-LABEL: func.func @specialized_size_int
// GENERALIZE: %[[DIM_NATIVE:[a-zA-Z0-9_]+]] = torchext.get %arg1 : !torch.int -> i64
// GENERALIZE: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.size %arg0[%[[DIM_NATIVE]]] : !torch.vtensor<[?,?],f32>
// GENERALIZE: torch_c.from_i64 %[[SIZE]]
func.func @specialized_size_int(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.int)
    -> !torch.int {
  %result = torch.aten.size.int %arg0, %arg1
      : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
  return %result : !torch.int
}

// GENERALIZE-LABEL: func.func @specialized_sub_int
// GENERALIZE: %[[SUB_A:[a-zA-Z0-9_]+]] = torchext.get %arg0 : !torch.int -> i64
// GENERALIZE: %[[SUB_B:[a-zA-Z0-9_]+]] = torchext.get %arg1 : !torch.int -> i64
// GENERALIZE: %[[SUB_RESULT:[a-zA-Z0-9_]+]] = arith.subi %[[SUB_A]], %[[SUB_B]] : i64
// GENERALIZE: torch_c.from_i64 %[[SUB_RESULT]]
func.func @specialized_sub_int(%arg0: !torch.int, %arg1: !torch.int)
    -> !torch.int {
  %result = torch.aten.sub.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}
