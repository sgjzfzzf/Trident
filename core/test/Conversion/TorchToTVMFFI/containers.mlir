//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// Frontend array accesses may temporarily carry their Torch element type.
// CHECK-LABEL: func.func @frontend_array_get_item(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.array) -> !tvm_ffi.tensor {
// CHECK: %[[INDEX:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[ITEM:[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item %[[ARRAY]][%[[INDEX]]] as !tvm_ffi.tensor
// CHECK: return %[[ITEM]] : !tvm_ffi.tensor
func.func @frontend_array_get_item(%arg0: !tvm_ffi.array)
    -> !torch.vtensor<[2,3],f32> {
  %index = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
  %item = tvm_ffi.array.get_item %arg0[%index]
      as !torch.vtensor<[2,3],f32>
      : !tvm_ffi.array, !tvm_ffi.int -> !torch.vtensor<[2,3],f32>
  return %item : !torch.vtensor<[2,3],f32>
}

// List and tuple containers share the TVM FFI array representation.
// CHECK-LABEL: func.func @container_construct(
// CHECK-SAME: -> !tvm_ffi.array {
// CHECK: "tvm_ffi.array.create"
func.func @container_construct(%arg0: !torch.int, %arg1: !torch.int)
    -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0, %arg1
      : (!torch.int, !torch.int) -> !torch.list<int>
  return %0 : !torch.list<int>
}

// CHECK-LABEL: func.func @tuple_construct(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[RHS:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-SAME: -> !tvm_ffi.array {
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]] = "tvm_ffi.array.create"(%[[LHS]], %[[RHS]])
// CHECK: return %[[TUPLE]] : !tvm_ffi.array
func.func @tuple_construct(%lhs: !torch.int, %rhs: !torch.int)
    -> !torch.tuple<int, int> {
  %tuple = torch.prim.TupleConstruct %lhs, %rhs
      : !torch.int, !torch.int -> !torch.tuple<int, int>
  return %tuple : !torch.tuple<int, int>
}

// CHECK-LABEL: func.func @list_unpack(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.array) -> (!tvm_ffi.int, !tvm_ffi.int) {
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[LHS:[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item %[[ARRAY]][%[[ZERO]]] as !tvm_ffi.int
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 1 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[RHS:[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item %[[ARRAY]][%[[ONE]]] as !tvm_ffi.int
// CHECK: return %[[LHS]], %[[RHS]] : !tvm_ffi.int, !tvm_ffi.int
func.func @list_unpack(%array: !torch.list<int>) -> (!torch.int, !torch.int) {
  %items:2 = torch.prim.ListUnpack %array
      : !torch.list<int> -> !torch.int, !torch.int
  return %items#0, %items#1 : !torch.int, !torch.int
}

// Container parameters are represented as one TVM FFI array at the ABI
// boundary. The conversion updates both the function signature and return.
// CHECK-LABEL: tvm_ffi.func @container_input(
// CHECK-SAME: %arg0: !tvm_ffi.array) -> !tvm_ffi.array {
// CHECK: tvm_ffi.return %arg0 : !tvm_ffi.array
tvm_ffi.func @container_input(
    %arg0: !torch.list<vtensor<[4],f32>>)
    -> !torch.list<vtensor<[4],f32>> {
  tvm_ffi.return %arg0 : !torch.list<vtensor<[4],f32>>
}
