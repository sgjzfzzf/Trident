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
// CHECK:         llvm.alloca {{%.*}} x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// Store the input operands.
// CHECK:         llvm.store %[[ARG0]], {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.store %[[ARG1]], {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// Call TVMFFIFunctionGetGlobal with the "trident.aten.mul.Scalar" name.
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