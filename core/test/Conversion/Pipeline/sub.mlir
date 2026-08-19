//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that aten.sub.Scalar and aten.sub.Tensor are lowered
// via the AtenGen FFI dispatch path: "trident.aten.sub.Scalar" and
// "trident.aten.sub.Tensor", called through independently cached TVM FFI
// function handles.

// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.sub.Scalar_trident.aten.sub.Scalar("trident.aten.sub.Scalar\00")
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.sub.Tensor_trident.aten.sub.Tensor("trident.aten.sub.Tensor\00")
// CHECK-LABEL:   llvm.func @torch.aten.sub.Scalar
// CHECK-SAME: %[[SCALAR_ARG0:.*]]: !llvm.struct<(i32, i32, i64)>, %[[SCALAR_ARG1:.*]]: !llvm.struct<(i32, i32, i64)>, %[[SCALAR_ARG2:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: %[[SCALAR_ARGS:.*]] = llvm.alloca %[[SCALAR_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[SCALAR_ARG0]], %[[SCALAR_ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[SCALAR_SLOT1:.*]] = llvm.getelementptr %[[SCALAR_ARGS]][1]
// CHECK: llvm.store %[[SCALAR_ARG1]], %[[SCALAR_SLOT1]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[SCALAR_SLOT2:.*]] = llvm.getelementptr %[[SCALAR_ARGS]][2]
// CHECK: llvm.store %[[SCALAR_ARG2]], %[[SCALAR_SLOT2]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: llvm.call @TVMFFIObjectDecRef
// CHECK: %[[SCALAR_RET:.*]] = llvm.load %[[SCALAR_RET_SLOT:.*]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @TVMFFIObjectIncRef
// CHECK: llvm.call @TVMFFIObjectDecRef
// CHECK: llvm.return %[[SCALAR_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL:   llvm.func @torch.aten.sub.Tensor
// CHECK-SAME: %[[TENSOR_ARG0:.*]]: !llvm.struct<(i32, i32, i64)>, %[[TENSOR_ARG1:.*]]: !llvm.struct<(i32, i32, i64)>, %[[TENSOR_ARG2:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: %[[TENSOR_ARGS:.*]] = llvm.alloca %[[TENSOR_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[TENSOR_ARG0]], %[[TENSOR_ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[TENSOR_SLOT1:.*]] = llvm.getelementptr %[[TENSOR_ARGS]][1]
// CHECK: llvm.store %[[TENSOR_ARG1]], %[[TENSOR_SLOT1]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[TENSOR_SLOT2:.*]] = llvm.getelementptr %[[TENSOR_ARGS]][2]
// CHECK: llvm.store %[[TENSOR_ARG2]], %[[TENSOR_SLOT2]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: llvm.call @TVMFFIObjectDecRef
// CHECK: %[[TENSOR_RET:.*]] = llvm.load %[[TENSOR_RET_SLOT:.*]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @TVMFFIObjectIncRef
// CHECK: llvm.call @TVMFFIObjectDecRef
// CHECK: llvm.return %[[TENSOR_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_sub_scalar(
// CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK: llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.getelementptr %arg1[2]
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: llvm.call @TVMFFIObjectDecRef
// CHECK: %[[WRAP_RET:.*]] = llvm.load %[[WRAP_RET_SLOT:.*]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @TVMFFIObjectIncRef
// CHECK: llvm.call @TVMFFIObjectDecRef
// CHECK: llvm.store %[[WRAP_RET]], %arg3


// Allocate the args array for 3 operands.
// Store the input operands.
// Build the function name struct {ptr, i64} and call TVMFFIFunctionGetGlobal.
// Load the function handle from the result slot.
// Set up the return value slot and call TVMFFIFunctionCall with the args.
// Release the function handle.
// Load the result TVMFFIAny from the return slot.
func.func @torch.aten.sub.Scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.sub.Scalar %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.float, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// Allocate the args array for 3 operands.
// Store the input operands.
// Build the function name struct {ptr, i64} and call TVMFFIFunctionGetGlobal.
// Load the function handle from the result slot.
// Set up the return value slot and call TVMFFIFunctionCall with the args.
// Release the function handle.
// Load the result TVMFFIAny from the return slot.
func.func @torch.aten.sub.Tensor(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

tvm_ffi.func @sub_scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.sub.Scalar %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.float, !torch.float -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

tvm_ffi.func @sub_tensor(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func internal @__trident_tvm_ffi_ctor_trident.aten.sub.Scalar() {
// CHECK:         %[[SCALAR_HANDLE_ADDR:.*]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.sub.Scalar : !llvm.ptr
// CHECK:         %[[SCALAR_NAME_LEN:.*]] = llvm.mlir.constant(23 : i64) : i64
// CHECK:         %[[SCALAR_NAME_PTR:.*]] = llvm.mlir.addressof @__trident_constant_trident.aten.sub.Scalar_trident.aten.sub.Scalar : !llvm.ptr
// CHECK:         %[[SCALAR_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[SCALAR_NAME_SLOT:.*]] = llvm.alloca %[[SCALAR_ONE]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// CHECK:         %[[SCALAR_NAME_PTR_SLOT:.*]] = llvm.getelementptr %[[SCALAR_NAME_SLOT]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK:         llvm.store %[[SCALAR_NAME_PTR]], %[[SCALAR_NAME_PTR_SLOT]] : !llvm.ptr, !llvm.ptr
// CHECK:         %[[SCALAR_NAME_LEN_SLOT:.*]] = llvm.getelementptr %[[SCALAR_NAME_SLOT]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK:         llvm.store %[[SCALAR_NAME_LEN]], %[[SCALAR_NAME_LEN_SLOT]] : i64, !llvm.ptr
// CHECK:         %[[SCALAR_FUNC_SLOT:.*]] = llvm.alloca %[[SCALAR_ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         %[[SCALAR_GET_GLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal(%[[SCALAR_NAME_SLOT]], %[[SCALAR_FUNC_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[SCALAR_HANDLE:.*]] = llvm.load %[[SCALAR_FUNC_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.store %[[SCALAR_HANDLE]], %[[SCALAR_HANDLE_ADDR]] : !llvm.ptr, !llvm.ptr
// CHECK:         llvm.return
// CHECK-LABEL: llvm.func internal @__trident_tvm_ffi_dtor_trident.aten.sub.Scalar() {
// CHECK:         %[[SCALAR_DTOR_ADDR:.*]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.sub.Scalar : !llvm.ptr
// CHECK:         %[[SCALAR_DTOR_HANDLE:.*]] = llvm.load %[[SCALAR_DTOR_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[SCALAR_DECREF:.*]] = llvm.call @TVMFFIObjectDecRef(%[[SCALAR_DTOR_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK:         llvm.return
// CHECK: llvm.mlir.global_ctors ctors = [@__trident_tvm_ffi_ctor_trident.aten.sub.Scalar], priorities = [65535 : i32], data = [#llvm.zero]
// CHECK: llvm.mlir.global_dtors dtors = [@__trident_tvm_ffi_dtor_trident.aten.sub.Scalar], priorities = [65535 : i32], data = [#llvm.zero]

// CHECK-LABEL: llvm.func internal @__trident_tvm_ffi_ctor_trident.aten.sub.Tensor() {
// CHECK:         %[[TENSOR_HANDLE_ADDR:.*]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.sub.Tensor : !llvm.ptr
// CHECK:         %[[TENSOR_NAME_LEN:.*]] = llvm.mlir.constant(23 : i64) : i64
// CHECK:         %[[TENSOR_NAME_PTR:.*]] = llvm.mlir.addressof @__trident_constant_trident.aten.sub.Tensor_trident.aten.sub.Tensor : !llvm.ptr
// CHECK:         %[[TENSOR_ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[TENSOR_NAME_SLOT:.*]] = llvm.alloca %[[TENSOR_ONE]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// CHECK:         %[[TENSOR_NAME_PTR_SLOT:.*]] = llvm.getelementptr %[[TENSOR_NAME_SLOT]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK:         llvm.store %[[TENSOR_NAME_PTR]], %[[TENSOR_NAME_PTR_SLOT]] : !llvm.ptr, !llvm.ptr
// CHECK:         %[[TENSOR_NAME_LEN_SLOT:.*]] = llvm.getelementptr %[[TENSOR_NAME_SLOT]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK:         llvm.store %[[TENSOR_NAME_LEN]], %[[TENSOR_NAME_LEN_SLOT]] : i64, !llvm.ptr
// CHECK:         %[[TENSOR_FUNC_SLOT:.*]] = llvm.alloca %[[TENSOR_ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         %[[TENSOR_GET_GLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal(%[[TENSOR_NAME_SLOT]], %[[TENSOR_FUNC_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[TENSOR_HANDLE:.*]] = llvm.load %[[TENSOR_FUNC_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.store %[[TENSOR_HANDLE]], %[[TENSOR_HANDLE_ADDR]] : !llvm.ptr, !llvm.ptr
// CHECK:         llvm.return
// CHECK-LABEL: llvm.func internal @__trident_tvm_ffi_dtor_trident.aten.sub.Tensor() {
// CHECK:         %[[TENSOR_DTOR_ADDR:.*]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.sub.Tensor : !llvm.ptr
// CHECK:         %[[TENSOR_DTOR_HANDLE:.*]] = llvm.load %[[TENSOR_DTOR_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[TENSOR_DECREF:.*]] = llvm.call @TVMFFIObjectDecRef(%[[TENSOR_DTOR_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK:         llvm.return
// CHECK: llvm.mlir.global_ctors ctors = [@__trident_tvm_ffi_ctor_trident.aten.sub.Tensor], priorities = [65535 : i32], data = [#llvm.zero]
// CHECK: llvm.mlir.global_dtors dtors = [@__trident_tvm_ffi_dtor_trident.aten.sub.Tensor], priorities = [65535 : i32], data = [#llvm.zero]
// CHECK-NOT: llvm.mlir.global_ctors
// CHECK-NOT: llvm.mlir.global_dtors
