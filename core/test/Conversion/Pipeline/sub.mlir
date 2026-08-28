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
// "trident.aten.sub.Tensor", called via TVMFFIFunctionGetGlobal /
// TVMFFIFunctionCall / TVMFFIObjectDecRef.

// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.sub.Scalar_trident.aten.sub.Scalar("trident.aten.sub.Scalar\00")
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.sub.Tensor_trident.aten.sub.Tensor("trident.aten.sub.Tensor\00")
// CHECK-LABEL:   llvm.func @torch.aten.sub.Scalar
// CHECK-SAME: %[[SCALAR_ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SCALAR_ARG1:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SCALAR_ARG2:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[SCALAR_FUNCTION_NAME:[0-9]+]], %[[SCALAR_HANDLE_SLOT:[0-9]+]])
// CHECK: %[[SCALAR_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[SCALAR_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[SCALAR_ARG0]], %[[SCALAR_ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[SCALAR_SLOT1:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SCALAR_ARGS]][1]
// CHECK: llvm.store %[[SCALAR_ARG1]], %[[SCALAR_SLOT1]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[SCALAR_SLOT2:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SCALAR_ARGS]][2]
// CHECK: llvm.store %[[SCALAR_ARG2]], %[[SCALAR_SLOT2]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[SCALAR_HANDLE:[0-9]+]], %[[SCALAR_CALL_ARGS:[0-9]+]], %[[SCALAR_ARG_COUNT:[0-9]+]], %[[SCALAR_RETURN_SLOT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[SCALAR_HANDLE]])
// CHECK: %[[SCALAR_RET:[a-zA-Z0-9_]+]] = llvm.load %[[SCALAR_RET_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[SCALAR_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL:   llvm.func @torch.aten.sub.Tensor
// CHECK-SAME: %[[TENSOR_ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[TENSOR_ARG1:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[TENSOR_ARG2:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[TENSOR_FUNCTION_NAME:[0-9]+]], %[[TENSOR_HANDLE_SLOT:[0-9]+]])
// CHECK: %[[TENSOR_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[TENSOR_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[TENSOR_ARG0]], %[[TENSOR_ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[TENSOR_SLOT1:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[TENSOR_ARGS]][1]
// CHECK: llvm.store %[[TENSOR_ARG1]], %[[TENSOR_SLOT1]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[TENSOR_SLOT2:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[TENSOR_ARGS]][2]
// CHECK: llvm.store %[[TENSOR_ARG2]], %[[TENSOR_SLOT2]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[TENSOR_HANDLE:[0-9]+]], %[[TENSOR_CALL_ARGS:[0-9]+]], %[[TENSOR_ARG_COUNT:[0-9]+]], %[[TENSOR_RETURN_SLOT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[TENSOR_HANDLE]])
// CHECK: %[[TENSOR_RET:[a-zA-Z0-9_]+]] = llvm.load %[[TENSOR_RET_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[TENSOR_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_sub_scalar(
// CHECK: llvm.call @sub_scalar

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

tvm_ffi.func @sub_scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.aten.sub.Scalar %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.float, !torch.float -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

tvm_ffi.func @sub_tensor(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}
