//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// CHECK-LABEL: func.func @transpose(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.t"
// CHECK: %[[CALL:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%[[ARG]]) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
// CHECK: return %[[CALL]] : !tvm_ffi.tensor
func.func @transpose(%arg0: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0
      : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// A value-semantic Torch clone may materialize a new memory layout before a
// raw-pointer kernel launch. It must reach the FFI lowering before Torch's
// folder, which ignores memory_format, can replace it with its input.
// CHECK-LABEL: func.func @clone_contiguous(
// CHECK: %[[FORMAT:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.clone"
// CHECK: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %[[FORMAT]]) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tensor
// CHECK: return %[[CLONE]] : !tvm_ffi.tensor
func.func @clone_contiguous(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> {
  %memory_format = torch.constant.int 0
  %0 = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  return %0 : !torch.vtensor<[32,2],f32>
}

// Non-ATen opaque operators are already legal for this conversion and must
// remain untouched.  ConvertAtenCall only handles generalized ATen names.
// CHECK-LABEL: func.func @opaque_operator() {
// CHECK-NEXT: torch.operator "torch.foo"() : () -> ()
// CHECK-NEXT: return
func.func @opaque_operator() {
  torch.operator "torch.foo"() : () -> ()
  return
}

// Multiple ATen results are represented by an FFI array and extracted in
// result order.  This is the semantic counterpart of the single-result ABI
// packing performed by the later TVMFFI transforms and TVMFFIToLLVM passes.
// CHECK-LABEL: func.func @multi_result(
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[KEEPDIM:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = false}> : () -> !tvm_ffi.bool
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.max.dim"
// CHECK: %[[PACKED:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %[[DIM]], %[[KEEPDIM]]) : (!tvm_ffi.tensor, !tvm_ffi.int, !tvm_ffi.bool) -> !tvm_ffi.array
// CHECK: %[[IDX0:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: tvm_ffi.array.get_item %[[PACKED]][%[[IDX0]]] as !tvm_ffi.tensor
// CHECK: %[[IDX1:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 1 : i64}> : () -> !tvm_ffi.int
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
