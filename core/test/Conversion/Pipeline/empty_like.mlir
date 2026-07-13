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
// CHECK:         %[[ARGS_ALLOCA:.*]] = llvm.alloca %[[ALLOCA_CNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// Store the input operand into the args array.
// CHECK:         llvm.store %[[ARG0]], %[[ARGS_ALLOCA]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
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
// IncRef + DecRef the returned object for correct ref-counting.
// CHECK:         %[[INC:.*]] = llvm.call @TVMFFIObjectIncRef(%[[OBJ_PTR:.*]]) : (!llvm.ptr) -> i32
// CHECK:         %[[DECREF_OBJ:.*]] = llvm.call @TVMFFIObjectDecRef(%[[OBJ_PTR2:.*]]) : (!llvm.ptr) -> i32
// CHECK:         llvm.return %[[RETLOAD]] : !llvm.struct<(i32, i32, i64)>
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
// CHECK:         llvm.br ^bb1
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.aten.empty_like(%[[LOADED]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.store %[[CALLEE_RET]], %[[WRAP_RET]]
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %0 = func.call @torch.aten.empty_like(%arg0) : (!torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64>
  tvm_ffi.return %0 : !torch.vtensor<[200,200,26],f64>
}
