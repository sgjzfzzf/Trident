//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops | FileCheck %s

// CHECK-LABEL: func.func @add_int
// CHECK: %[[ADD_A:[a-zA-Z0-9_]+]] = torchext.get %arg0 : !torch.int -> i64
// CHECK: %[[ADD_B:[a-zA-Z0-9_]+]] = torchext.get %arg1 : !torch.int -> i64
// CHECK: %[[ADD_RESULT:[a-zA-Z0-9_]+]] = arith.addi %[[ADD_A]], %[[ADD_B]] : i64
// CHECK: %[[ADD_INT:[a-zA-Z0-9_]+]] = torch_c.from_i64 %[[ADD_RESULT]]
func.func @add_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %result = torch.aten.add.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @floordiv_int
// CHECK: %[[FLOOR_A:[a-zA-Z0-9_]+]] = torchext.get %arg0 : !torch.int -> i64
// CHECK: %[[FLOOR_B:[a-zA-Z0-9_]+]] = torchext.get %arg1 : !torch.int -> i64
// CHECK: %[[FLOOR_RESULT:[a-zA-Z0-9_]+]] = arith.floordivsi %[[FLOOR_A]], %[[FLOOR_B]] : i64
// CHECK: torch_c.from_i64 %[[FLOOR_RESULT]]
func.func @floordiv_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %result = torch.aten.floordiv.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @int_bool
// CHECK: %[[BOOL_NATIVE:[a-zA-Z0-9_]+]] = torchext.get %arg0 : !torch.bool -> i1
// CHECK: %[[BOOL_INT:[a-zA-Z0-9_]+]] = arith.extui %[[BOOL_NATIVE]] : i1 to i64
// CHECK: torch_c.from_i64 %[[BOOL_INT]]
func.func @int_bool(%arg0: !torch.bool) -> !torch.int {
  %result = torch.aten.Int.bool %arg0 : !torch.bool -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @size_int
// CHECK: %[[DIM_NATIVE:[a-zA-Z0-9_]+]] = torchext.get %arg1 : !torch.int -> i64
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.size %arg0[%[[DIM_NATIVE]]] : !torch.vtensor<[?,?],f32>
// CHECK: torch_c.from_i64 %[[SIZE]]
func.func @size_int(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.int)
    -> !torch.int {
  %result = torch.aten.size.int %arg0, %arg1
      : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @sub_int
// CHECK: %[[SUB_A:[a-zA-Z0-9_]+]] = torchext.get %arg0 : !torch.int -> i64
// CHECK: %[[SUB_B:[a-zA-Z0-9_]+]] = torchext.get %arg1 : !torch.int -> i64
// CHECK: %[[SUB_RESULT:[a-zA-Z0-9_]+]] = arith.subi %[[SUB_A]], %[[SUB_B]] : i64
// CHECK: torch_c.from_i64 %[[SUB_RESULT]]
func.func @sub_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %result = torch.aten.sub.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}
