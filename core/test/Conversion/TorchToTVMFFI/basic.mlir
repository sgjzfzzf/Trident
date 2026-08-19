//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-torch-to-tvm-ffi | FileCheck %s

func.func @transpose(%arg0: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[3,2],f32> {
  // CHECK-LABEL: func.func @transpose(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  // CHECK: %[[FUNC:.*]] = tvm_ffi.FunctionGetGlobal "trident.aten.t"
  // CHECK: %[[CALL:.*]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0)
  // CHECK-SAME: -> !tvm_ffi.tensor
  // CHECK: tvm_ffi.ObjectIncRef %[[CALL]]
  // CHECK: tvm_ffi.ObjectDecRef %[[CALL]]
  // CHECK: return %[[CALL]] : !tvm_ffi.tensor
  %0 = torch.aten.t %arg0
      : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// Constants are converted to their semantic TVM FFI scalar types, rather than
// being lowered through the LLVM representation used by the final ABI pass.
// CHECK-LABEL: func.func @constants(
// CHECK-SAME: -> !tvm_ffi.array {
// CHECK: %[[I0:.*]] = "tvm_ffi.constant"() <{value = 7 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[I1:.*]] = "tvm_ffi.constant"() <{value = 9 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[ARRAY:.*]] = "tvm_ffi.array.create"(%[[I0]], %[[I1]])
// CHECK: tvm_ffi.ObjectIncRef %[[ARRAY]] : !tvm_ffi.array
// CHECK: tvm_ffi.ObjectDecRef %[[ARRAY]] : !tvm_ffi.array
// CHECK: return %[[ARRAY]] : !tvm_ffi.array
func.func @constants() -> !torch.list<int> {
  %i = torch.constant.int 7
  %j = torch.constant.int 9
  %array = torch.prim.ListConstruct %i, %j
      : (!torch.int, !torch.int) -> !torch.list<int>
  return %array : !torch.list<int>
}

// CHECK-LABEL: func.func @constant_float() -> !tvm_ffi.float {
// CHECK: %[[FLOAT:.*]] = "tvm_ffi.constant"() <{value = 2.500000e+00 : f64}> : () -> !tvm_ffi.float
// CHECK: return %[[FLOAT]] : !tvm_ffi.float
func.func @constant_float() -> !torch.float {
  %0 = torch.constant.float 2.5
  return %0 : !torch.float
}

// CHECK-LABEL: func.func @constant_bool() -> !tvm_ffi.bool {
// CHECK: %[[BOOL:.*]] = "tvm_ffi.constant"() <{value = true}> : () -> !tvm_ffi.bool
// CHECK: return %[[BOOL]] : !tvm_ffi.bool
func.func @constant_bool() -> !torch.bool {
  %0 = torch.constant.bool true
  return %0 : !torch.bool
}

// CHECK-LABEL: func.func @constant_none() -> !tvm_ffi.none {
// CHECK: %[[NONE:.*]] = "tvm_ffi.constant"() <{value}> : () -> !tvm_ffi.none
// CHECK: return %[[NONE]] : !tvm_ffi.none
func.func @constant_none() -> !torch.none {
  %0 = torch.constant.none
  return %0 : !torch.none
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

// Multiple ATen results are represented by an FFI array and extracted in
// result order.  This is the semantic counterpart of the single-result ABI
// packing performed by the later DecomposeTVMFFI/TVMFFIToLLVM passes.
// CHECK-LABEL: func.func @multi_result(
// CHECK: %[[FUNC:.*]] = tvm_ffi.FunctionGetGlobal "trident.aten.max.dim"
// CHECK: %[[PACKED:.*]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %[[DIM:.*]], %[[KEEPDIM:.*]]) : (!tvm_ffi.tensor, !tvm_ffi.int, !tvm_ffi.bool) -> !tvm_ffi.array
// CHECK: %[[IDX0:.*]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: tvm_ffi.array.get_item %[[PACKED]][%[[IDX0]]] as !tvm_ffi.tensor
// CHECK: %[[IDX1:.*]] = "tvm_ffi.constant"() <{value = 1 : i64}> : () -> !tvm_ffi.int
// CHECK: tvm_ffi.array.get_item %[[PACKED]][%[[IDX1]]] as !tvm_ffi.tensor
func.func @multi_result(%arg0: !torch.vtensor<[4],f32>)
    -> (!torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>) {
  %dim = torch.constant.int 0
  %keepdim = torch.constant.bool false
  %0, %1 = torch.aten.max.dim %arg0, %dim, %keepdim
      : !torch.vtensor<[4],f32>, !torch.int, !torch.bool
      -> !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>
  return %0, %1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>
}

// Each branch of torch.prim.If is processed with its own Region-local Set.
// The tensor produced by torch.aten.t is IncRef'd for the branch yield and
// DecRef'd for the branch-local ownership.
// CHECK-LABEL: func.func @nested_if
// CHECK: torch.prim.If
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: torch.prim.If.yield
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: torch.prim.If.yield
func.func @nested_if(%arg0: !torch.vtensor<[2,3],f32>, %cond: !torch.bool)
    -> !torch.vtensor<[3,2],f32> {
  %0 = torch.prim.If %cond -> (!torch.vtensor<[3,2],f32>) {
    %1 = torch.aten.t %arg0
        : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    torch.prim.If.yield %1 : !torch.vtensor<[3,2],f32>
  } else {
    %2 = torch.aten.t %arg0
        : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    torch.prim.If.yield %2 : !torch.vtensor<[3,2],f32>
  }
  return %0 : !torch.vtensor<[3,2],f32>
}

// The TVMFFI function body is also a Region root, so its return is handled
// by the same terminator pattern.
// CHECK-LABEL: tvm_ffi.func @ffi_return
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: tvm_ffi.return
tvm_ffi.func @ffi_return(%arg0: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0
      : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  tvm_ffi.return %0 : !torch.vtensor<[3,2],f32>
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
