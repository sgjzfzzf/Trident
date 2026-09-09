//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// List and tuple containers share the TVM FFI array representation.
// CHECK-LABEL: func.func @container_construct(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[RHS:[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.array {
// CHECK: %[[FUNCTION:[a-zA-Z0-9_]+]], %[[GET_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[GET_SUCCESS]], "TVMFFIFunctionGetGlobal failed for ffi.Array"
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]], %[[CALL_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNCTION]](%[[LHS]], %[[RHS]]) : (!tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.array, i1
// CHECK-NEXT: cf.assert %[[CALL_SUCCESS]], "TVMFFIFunctionCall failed for ffi.Array"
// CHECK-NEXT: return %[[ARRAY]] : !tvm_ffi.array
func.func @container_construct(%arg0: !torch.int, %arg1: !torch.int)
    -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0, %arg1
      : (!torch.int, !torch.int) -> !torch.list<int>
  return %0 : !torch.list<int>
}

// CHECK-LABEL: func.func @tuple_construct(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[RHS:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK-SAME: -> !tvm_ffi.array {
// CHECK: %[[FUNCTION:[a-zA-Z0-9_]+]], %[[GET_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[GET_SUCCESS]], "TVMFFIFunctionGetGlobal failed for ffi.Array"
// CHECK: %[[TUPLE:[a-zA-Z0-9_]+]], %[[CALL_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNCTION]](%[[LHS]], %[[RHS]]) : (!tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.array, i1
// CHECK-NEXT: cf.assert %[[CALL_SUCCESS]], "TVMFFIFunctionCall failed for ffi.Array"
// CHECK: return %[[TUPLE]] : !tvm_ffi.array
func.func @tuple_construct(%lhs: !torch.int, %rhs: !torch.int)
    -> !torch.tuple<int, int> {
  %tuple = torch.prim.TupleConstruct %lhs, %rhs
      : !torch.int, !torch.int -> !torch.tuple<int, int>
  return %tuple : !torch.tuple<int, int>
}

// CHECK-LABEL: func.func @list_unpack(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.array) -> (!tvm_ffi.int, !tvm_ffi.int) {
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 0
// CHECK-NEXT: %[[LHS_FUNCTION:[a-zA-Z0-9_]+]], %[[LHS_GET_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[LHS_GET_SUCCESS]], "TVMFFIFunctionGetGlobal failed for ffi.ArrayGetItem"
// CHECK: %[[LHS:[a-zA-Z0-9_]+]], %[[LHS_CALL_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[LHS_FUNCTION]](%[[ARRAY]], %[[ZERO]]) : (!tvm_ffi.array, !tvm_ffi.int) -> !tvm_ffi.int, i1
// CHECK-NEXT: cf.assert %[[LHS_CALL_SUCCESS]], "TVMFFIFunctionCall failed for ffi.ArrayGetItem"
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK-NEXT: %[[RHS_FUNCTION:[a-zA-Z0-9_]+]], %[[RHS_GET_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[RHS_GET_SUCCESS]], "TVMFFIFunctionGetGlobal failed for ffi.ArrayGetItem"
// CHECK: %[[RHS:[a-zA-Z0-9_]+]], %[[RHS_CALL_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[RHS_FUNCTION]](%[[ARRAY]], %[[ONE]]) : (!tvm_ffi.array, !tvm_ffi.int) -> !tvm_ffi.int, i1
// CHECK-NEXT: cf.assert %[[RHS_CALL_SUCCESS]], "TVMFFIFunctionCall failed for ffi.ArrayGetItem"
// CHECK: return %[[LHS]], %[[RHS]] : !tvm_ffi.int, !tvm_ffi.int
func.func @list_unpack(%array: !torch.list<int>) -> (!torch.int, !torch.int) {
  %items:2 = torch.prim.ListUnpack %array
      : !torch.list<int> -> !torch.int, !torch.int
  return %items#0, %items#1 : !torch.int, !torch.int
}

// Container parameters are represented as one TVM FFI array at the ABI
// boundary. The conversion updates both the function signature and return.
// CHECK-LABEL: tvm_ffi.func @container_input(
// CHECK-SAME: %[[ARRAY_INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.array) -> !tvm_ffi.array {
// CHECK: tvm_ffi.return %[[ARRAY_INPUT]] : !tvm_ffi.array
tvm_ffi.func @container_input(
    %arg0: !torch.list<vtensor<[4],f32>>)
    -> !torch.list<vtensor<[4],f32>> {
  tvm_ffi.return %arg0 : !torch.list<vtensor<[4],f32>>
}
