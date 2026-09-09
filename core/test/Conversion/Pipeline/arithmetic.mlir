//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// CHECK-LABEL: llvm.func @torch.aten.mul.Scalar
// CHECK-SAME: %[[MUL_ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[MUL_ARG1:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[MUL_NAME:[a-zA-Z0-9_]+]], %[[MUL_HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[MUL_HANDLE:[a-zA-Z0-9_]+]], %[[MUL_ARGS:[a-zA-Z0-9_]+]], %[[MUL_NARGS:[a-zA-Z0-9_]+]], %[[MUL_RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[MUL_RET:[a-zA-Z0-9_]+]] = llvm.load %[[MUL_RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[MUL_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @torch.aten.sub.Scalar
// CHECK-SAME: %[[SUB_SCALAR_ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SUB_SCALAR_ARG1:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SUB_SCALAR_ARG2:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[SUB_SCALAR_NAME:[a-zA-Z0-9_]+]], %[[SUB_SCALAR_HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[SUB_SCALAR_HANDLE:[a-zA-Z0-9_]+]], %[[SUB_SCALAR_ARGS:[a-zA-Z0-9_]+]], %[[SUB_SCALAR_NARGS:[a-zA-Z0-9_]+]], %[[SUB_SCALAR_RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[SUB_SCALAR_RET:[a-zA-Z0-9_]+]] = llvm.load %[[SUB_SCALAR_RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[SUB_SCALAR_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @torch.aten.sub.Tensor
// CHECK-SAME: %[[SUB_TENSOR_ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SUB_TENSOR_ARG1:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SUB_TENSOR_ARG2:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[SUB_TENSOR_NAME:[a-zA-Z0-9_]+]], %[[SUB_TENSOR_HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[SUB_TENSOR_HANDLE:[a-zA-Z0-9_]+]], %[[SUB_TENSOR_ARGS:[a-zA-Z0-9_]+]], %[[SUB_TENSOR_NARGS:[a-zA-Z0-9_]+]], %[[SUB_TENSOR_RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[SUB_TENSOR_RET:[a-zA-Z0-9_]+]] = llvm.load %[[SUB_TENSOR_RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[SUB_TENSOR_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_mul_scalar(
// CHECK: llvm.call @mul_scalar(%[[MUL_WRAPPER_INPUT:[a-zA-Z0-9_]+]], %[[MUL_WRAPPER_SCALAR:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_sub_scalar(
// CHECK: llvm.call @sub_scalar(%[[SUB_SCALAR_WRAPPER_INPUT:[a-zA-Z0-9_]+]], %[[SUB_SCALAR_WRAPPER_OTHER:[a-zA-Z0-9_]+]], %[[SUB_SCALAR_WRAPPER_ALPHA:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_sub_tensor(
// CHECK: llvm.call @sub_tensor(%[[SUB_TENSOR_WRAPPER_INPUT:[a-zA-Z0-9_]+]], %[[SUB_TENSOR_WRAPPER_OTHER:[a-zA-Z0-9_]+]], %[[SUB_TENSOR_WRAPPER_ALPHA:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>

func.func @torch.aten.mul.Scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mul.Scalar %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

func.func @torch.aten.sub.Scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.sub.Scalar %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.float, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

func.func @torch.aten.sub.Tensor(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

tvm_ffi.func @mul_scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float) -> !torch.vtensor<[2,3],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.aten.mul.Scalar %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

tvm_ffi.func @sub_scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.aten.sub.Scalar %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.float, !torch.float -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

tvm_ffi.func @sub_tensor(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}
