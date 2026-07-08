//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that torch.aten.empty_like is lowered via the AtenGen
// FFI dispatch path: "trident.aten.empty_like", called via
// TVMFFIFunctionGetGlobal / TVMFFIFunctionCall / TVMFFIObjectDecRef.

// CHECK-DAG: llvm.func @TVMFFIObjectIncRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32

// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.empty_like_trident.aten.empty_like("trident.aten.empty_like\00")

// CHECK-LABEL: llvm.func @torch.aten.empty_like(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// Allocate the args array for 6 operands.
// CHECK:         llvm.alloca {{%.*}} x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// Store the input operand.
// CHECK:         llvm.store %[[ARG0]], {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// Call TVMFFIFunctionGetGlobal with the "trident.aten.empty_like" name.
// CHECK:         llvm.call @TVMFFIFunctionGetGlobal
// Call TVMFFIFunctionCall with the handle.
// CHECK:         llvm.call @TVMFFIFunctionCall
// DecRef the function handle.
// CHECK:         llvm.call @TVMFFIObjectDecRef
// Load the result TVMFFIAny.
// CHECK:         llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// IncRef + DecRef the returned object.
// CHECK:         llvm.call @TVMFFIObjectIncRef
// CHECK:         llvm.call @TVMFFIObjectDecRef
// CHECK:         llvm.return {{%.*}} : !llvm.struct<(i32, i32, i64)>
func.func @torch.aten.empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return %0 : !torch.vtensor<[200,200,26],f64>
}

// tvm_ffi.func wrapper: calls func.func @torch.aten.empty_like via func.call.
// CHECK-LABEL: llvm.func @__tvm_ffi_empty_like(
// CHECK-SAME:    %arg0: !llvm.ptr, %[[WRAP_ARGS:.*]]: !llvm.ptr, %arg2: i32, %[[WRAP_RET:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[LOADED:.*]] = llvm.load %[[WRAP_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.br
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.aten.empty_like({{%.*}}) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.store %[[CALLEE_RET]], %[[WRAP_RET]]
// CHECK:         llvm.return %[[ZERO]] : i32

tvm_ffi.func @empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %0 = func.call @torch.aten.empty_like(%arg0) : (!torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64>
  tvm_ffi.return %0 : !torch.vtensor<[200,200,26],f64>
}
