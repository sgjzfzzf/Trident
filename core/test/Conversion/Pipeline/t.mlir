//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that torch.aten.t (transpose view) is lowered via the
// AtenGen FFI dispatch path: "trident.aten.t", called via
// TVMFFIFunctionGetGlobal / TVMFFIFunctionCall / TVMFFIObjectDecRef.
//
// NOTE: aten.t is a view op — its result aliases the operand's storage. This
// test only checks that the FFI dispatch path is generated correctly; the
// runtime semantics of the transposed view (strides preserved across the
// DLPack boundary) are exercised end-to-end by test/test_t.py.

// CHECK-DAG: llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32

// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.t_trident.aten.t("trident.aten.t\00")

// CHECK-LABEL: llvm.func @torch.aten.t(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// Allocate the args array for 1 operand.
// CHECK:         %[[ARRAY:.*]] = llvm.alloca %[[CNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// Store the input operand.
// CHECK:         llvm.store %[[ARG0]], %[[ARRAY]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// Build the function name struct {ptr, i64} and call TVMFFIFunctionGetGlobal.
// CHECK:         %[[GETGLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal(%[[NAME_SLOT:.*]], %[[HANDLE_SLOT:.*]]) : (!llvm.ptr, !llvm.ptr) -> i32
// Load the function handle from the result slot.
// CHECK:         %[[HANDLE:.*]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// Set up the return value slot and call TVMFFIFunctionCall with the args.
// CHECK:         %[[FUNCCALL:.*]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS_COPY:.*]], %[[NARGS:.*]], %[[RET_SLOT:.*]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// Release the function handle.
// CHECK:         %[[DECREF_HANDLE:.*]] = llvm.call @TVMFFIObjectDecRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
// Load the result TVMFFIAny from the return slot.
// CHECK:         %[[RETLOAD:.*]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.return %[[RETLOAD]] : !llvm.struct<(i32, i32, i64)>
func.func @torch.aten.t(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_t(
// CHECK-SAME:    %arg0: !llvm.ptr, %[[ARGS:.*]]: !llvm.ptr, %arg2: i32, %[[RET:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[TENSOR_LOAD:.*]] = llvm.load %[[ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.aten.t(%[[TENSOR_LOAD]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.store %[[CALLEE_RET]], %[[RET]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @t(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32> {
  %0 = func.call @torch.aten.t(%arg0) : (!torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32>
  tvm_ffi.return %0 : !torch.vtensor<[3,2],f32>
}
