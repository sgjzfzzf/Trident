//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that torch.aten.mul.Scalar is lowered via the AtenGen
// FFI dispatch path: "trident.aten.mul.Scalar", called via
// TVMFFIFunctionGetGlobal / TVMFFIFunctionCall / TVMFFIObjectDecRef.

// CHECK-DAG: llvm.func @TVMFFIObjectIncRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32

// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.mul.Scalar_trident.aten.mul.Scalar("trident.aten.mul.Scalar\00")

// CHECK-LABEL: llvm.func @torch.aten.mul.Scalar(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.struct<(i32, i32, i64)>, %[[ARG1:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// Allocate the args array for 2 operands.
// CHECK:         %[[ARRAY:.*]] = llvm.alloca %[[CNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// Store the input operands.
// CHECK:         llvm.store %[[ARG0]], %[[ARRAY]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.store %[[ARG1]], {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
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
func.func @torch.aten.mul.Scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mul.Scalar %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_mul_scalar(
// CHECK-SAME:    %arg0: !llvm.ptr, %[[ARGS:.*]]: !llvm.ptr, %arg2: i32, %[[RET:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[TENSOR_LOAD:.*]] = llvm.load %[[ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_GEP:.*]] = llvm.getelementptr %[[ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_LOAD:.*]] = llvm.load %[[SCALAR_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.aten.mul.Scalar(%[[TENSOR_LOAD]], %[[SCALAR_LOAD]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.store %[[CALLEE_RET]], %[[RET]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @mul_scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = func.call @torch.aten.mul.Scalar(%arg0, %arg1) : (!torch.vtensor<[2,3],f32>, !torch.float) -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}