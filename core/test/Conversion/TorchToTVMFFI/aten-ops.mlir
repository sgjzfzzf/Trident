//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// CHECK-LABEL: func.func @transpose(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]], %[[GET_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.t" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[GET_SUCCESS]], "TVMFFIFunctionGetGlobal failed for trident.aten.t"
// CHECK-NEXT: %[[FUNCTION_PTR:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[FUNC]] : !tvm_ffi.function to !llvm.ptr
// CHECK-NEXT: %[[NULL:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[NON_NULL:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %[[FUNCTION_PTR]], %[[NULL]] : !llvm.ptr
// CHECK-NEXT: cf.assert %[[NON_NULL]], "TVMFFIFunctionGetGlobal returned null for trident.aten.t"
// CHECK-NEXT: %[[CALL:[a-zA-Z0-9_]+]], %[[CALL_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%[[ARG]]) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor, i1
// CHECK-NEXT: cf.assert %[[CALL_SUCCESS]], "TVMFFIFunctionCall failed for trident.aten.t"
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
// CHECK: %[[FORMAT:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 0
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]], %[[GET_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.clone" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[GET_SUCCESS]], "TVMFFIFunctionGetGlobal failed for trident.aten.clone"
// CHECK: %[[CLONE:[a-zA-Z0-9_]+]], %[[CALL_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %[[FORMAT]]) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tensor, i1
// CHECK-NEXT: cf.assert %[[CALL_SUCCESS]], "TVMFFIFunctionCall failed for trident.aten.clone"
// CHECK: return %[[CLONE]] : !tvm_ffi.tensor
func.func @clone_contiguous(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> {
  %memory_format = torch.constant.int 0
  %0 = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  return %0 : !torch.vtensor<[32,2],f32>
}

// Multiple ATen results are represented by an FFI array and extracted in
// result order.  This is the semantic counterpart of the single-result ABI
// packing performed by the later TVMFFI transforms and TVMFFIToLLVM passes.
// CHECK-LABEL: func.func @multi_result(
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 0
// CHECK: %[[KEEPDIM:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool false
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]], %[[GET_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.max.dim" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[GET_SUCCESS]], "TVMFFIFunctionGetGlobal failed for trident.aten.max.dim"
// CHECK: %[[PACKED:[a-zA-Z0-9_]+]], %[[CALL_SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %[[DIM]], %[[KEEPDIM]]) : (!tvm_ffi.tensor, !tvm_ffi.int, !tvm_ffi.bool) -> !tvm_ffi.array, i1
// CHECK-NEXT: cf.assert %[[CALL_SUCCESS]], "TVMFFIFunctionCall failed for trident.aten.max.dim"
// CHECK: %[[IDX0:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 0
// CHECK-NEXT: %[[ITEM_FUNC0:[a-zA-Z0-9_]+]], %[[ITEM_GET_SUCCESS0:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[ITEM_GET_SUCCESS0]], "TVMFFIFunctionGetGlobal failed for ffi.ArrayGetItem"
// CHECK: %[[VALUE0:[a-zA-Z0-9_]+]], %[[ITEM_CALL_SUCCESS0:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[ITEM_FUNC0]](%[[PACKED]], %[[IDX0]]) : (!tvm_ffi.array, !tvm_ffi.int) -> !tvm_ffi.tensor, i1
// CHECK-NEXT: cf.assert %[[ITEM_CALL_SUCCESS0]], "TVMFFIFunctionCall failed for ffi.ArrayGetItem"
// CHECK: %[[IDX1:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// CHECK-NEXT: %[[ITEM_FUNC1:[a-zA-Z0-9_]+]], %[[ITEM_GET_SUCCESS1:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.function, i1
// CHECK-NEXT: cf.assert %[[ITEM_GET_SUCCESS1]], "TVMFFIFunctionGetGlobal failed for ffi.ArrayGetItem"
// CHECK: %[[VALUE1:[a-zA-Z0-9_]+]], %[[ITEM_CALL_SUCCESS1:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[ITEM_FUNC1]](%[[PACKED]], %[[IDX1]]) : (!tvm_ffi.array, !tvm_ffi.int) -> !tvm_ffi.tensor, i1
// CHECK-NEXT: cf.assert %[[ITEM_CALL_SUCCESS1]], "TVMFFIFunctionCall failed for ffi.ArrayGetItem"
// CHECK: return %[[VALUE0]], %[[VALUE1]] : !tvm_ffi.tensor, !tvm_ffi.tensor
func.func @multi_result(%arg0: !torch.vtensor<[4],f32>)
    -> (!torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>) {
  %dim = torch.constant.int 0
  %keepdim = torch.constant.bool false
  %0, %1 = torch.aten.max.dim %arg0, %dim, %keepdim
      : !torch.vtensor<[4],f32>, !torch.int, !torch.bool
      -> !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>
  return %0, %1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>
}
