//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @clone(
// CHECK-SAME: %[[CLONE_INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[CLONE_SIZE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: %[[CLONE_INPUT_SLOT:[0-9]+]] = llvm.alloca %[[CLONE_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK-NEXT: llvm.store %[[CLONE_INPUT]], %[[CLONE_INPUT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[CLONE_OUTPUT_SIZE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[CLONE_OUTPUT_SLOT:[0-9]+]] = llvm.alloca %[[CLONE_OUTPUT_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[CLONE_OUTPUT_PTR:[0-9]+]] = llvm.getelementptr %[[CLONE_OUTPUT_SLOT]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: %[[CLONE_INPUT_VALUE:[0-9]+]] = llvm.load %[[CLONE_INPUT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[CLONE_INPUT_VALUE]], %[[CLONE_OUTPUT_PTR]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[CLONE_NUM_ARGS:[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[CLONE_STRING_SIZE:[0-9]+]] = llvm.mlir.constant(29 : i64) : i64
// CHECK: %[[CLONE_GLOBAL_NAME:[0-9]+]] = llvm.alloca %[[CLONE_STRING_SIZE]] x i8 : (i64) -> !llvm.ptr
// CHECK: %[[CLONE_GLOBAL_NAME_SIZE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[CLONE_GLOBAL_NAME_SLOT:[0-9]+]] = llvm.alloca %[[CLONE_GLOBAL_NAME_SIZE]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[CLONE_GLOBAL_NAME_PTR:[0-9]+]] = llvm.getelementptr %[[CLONE_GLOBAL_NAME_SLOT]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK: llvm.store %[[CLONE_GLOBAL_NAME]], %[[CLONE_GLOBAL_NAME_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[CLONE_GLOBAL_NAME_LENGTH:[0-9]+]] = llvm.mlir.constant(28 : i64) : i64
// CHECK: %[[CLONE_GLOBAL_LENGTH_PTR:[0-9]+]] = llvm.getelementptr %[[CLONE_GLOBAL_NAME_SLOT]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK: llvm.store %[[CLONE_GLOBAL_NAME_LENGTH]], %[[CLONE_GLOBAL_LENGTH_PTR]] : i64, !llvm.ptr
// CHECK: %[[CLONE_HANDLE_SIZE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[CLONE_HANDLE_SLOT:[0-9]+]] = llvm.alloca %[[CLONE_HANDLE_SIZE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK: %[[CLONE_GLOBAL_STATUS:[0-9]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[CLONE_GLOBAL_NAME_SLOT]], %[[CLONE_HANDLE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[CLONE_HANDLE:[0-9]+]] = llvm.load %[[CLONE_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[CLONE_CALL_ARGS_SIZE:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[CLONE_CALL_ARGS_COUNT:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[CLONE_CALL_ARGS:[0-9]+]] = llvm.alloca %[[CLONE_CALL_ARGS_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[CLONE_CALL_ARGS_KIND_PTR:[0-9]+]] = llvm.getelementptr %[[CLONE_CALL_ARGS]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[CLONE_CALL_ARGS_SIZE]], %[[CLONE_CALL_ARGS_KIND_PTR]] : i32, !llvm.ptr
// CHECK: %[[CLONE_CALL_ARGS_DEVICE_PTR:[0-9]+]] = llvm.getelementptr %[[CLONE_CALL_ARGS]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[CLONE_CALL_ARGS_SIZE]], %[[CLONE_CALL_ARGS_DEVICE_PTR]] : i32, !llvm.ptr
// CHECK: %[[CLONE_CALL_ARGS_HANDLE:[0-9]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[CLONE_CALL_ARGS_HANDLE_PTR:[0-9]+]] = llvm.getelementptr %[[CLONE_CALL_ARGS]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[CLONE_CALL_ARGS_HANDLE]], %[[CLONE_CALL_ARGS_HANDLE_PTR]] : i64, !llvm.ptr
// CHECK: %[[CLONE_CALL_STATUS:[0-9]+]] = llvm.call @TVMFFIFunctionCall(%[[CLONE_HANDLE]], %[[CLONE_OUTPUT_SLOT]], %[[CLONE_NUM_ARGS]], %[[CLONE_CALL_ARGS]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[CLONE_RESULT:[0-9]+]] = llvm.load %[[CLONE_CALL_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[CLONE_RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @clone(%input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %copy = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  return %copy : !tvm_ffi.tensor
}

// CHECK-LABEL: func.func @copy(
// CHECK-SAME: %[[COPY_DESTINATION:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[COPY_SOURCE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK: %[[COPY_SIZE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[COPY_DEST_SLOT:[0-9]+]] = llvm.alloca %[[COPY_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[COPY_SOURCE_SLOT:[0-9]+]] = llvm.alloca %[[COPY_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[COPY_DESTINATION]], %[[COPY_DEST_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.store %[[COPY_SOURCE]], %[[COPY_SOURCE_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[COPY_ARGUMENTS_COUNT:[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[COPY_ARGUMENTS_SLOT:[0-9]+]] = llvm.alloca %[[COPY_ARGUMENTS_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[COPY_NUM_ARGS:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[COPY_STRING_SIZE:[0-9]+]] = llvm.mlir.constant(29 : i64) : i64
// CHECK: %[[COPY_GLOBAL_NAME:[0-9]+]] = llvm.alloca %[[COPY_STRING_SIZE]] x i8 : (i64) -> !llvm.ptr
// CHECK: %[[COPY_GLOBAL_NAME_SIZE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[COPY_GLOBAL_NAME_SLOT:[0-9]+]] = llvm.alloca %[[COPY_GLOBAL_NAME_SIZE]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[COPY_GLOBAL_NAME_PTR:[0-9]+]] = llvm.getelementptr %[[COPY_GLOBAL_NAME_SLOT]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK: llvm.store %[[COPY_GLOBAL_NAME]], %[[COPY_GLOBAL_NAME_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[COPY_GLOBAL_NAME_LENGTH:[0-9]+]] = llvm.mlir.constant(28 : i64) : i64
// CHECK: %[[COPY_GLOBAL_LENGTH_PTR:[0-9]+]] = llvm.getelementptr %[[COPY_GLOBAL_NAME_SLOT]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK: llvm.store %[[COPY_GLOBAL_NAME_LENGTH]], %[[COPY_GLOBAL_LENGTH_PTR]] : i64, !llvm.ptr
// CHECK: %[[COPY_HANDLE_SIZE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[COPY_HANDLE_SLOT:[0-9]+]] = llvm.alloca %[[COPY_HANDLE_SIZE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK: %[[COPY_GETGLOBAL_STATUS:[0-9]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[COPY_GLOBAL_NAME_SLOT]], %[[COPY_HANDLE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[COPY_HANDLE:[0-9]+]] = llvm.load %[[COPY_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[COPY_CALL_ARGS_KIND:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[COPY_CALL_ARGS_SIZE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[COPY_CALL_ARGS:[0-9]+]] = llvm.alloca %[[COPY_CALL_ARGS_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[COPY_CALL_ARGS_KIND_PTR:[0-9]+]] = llvm.getelementptr %[[COPY_CALL_ARGS]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY_CALL_ARGS_KIND]], %[[COPY_CALL_ARGS_KIND_PTR]] : i32, !llvm.ptr
// CHECK: %[[COPY_CALL_ARGS_DEVICE_PTR:[0-9]+]] = llvm.getelementptr %[[COPY_CALL_ARGS]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY_CALL_ARGS_KIND]], %[[COPY_CALL_ARGS_DEVICE_PTR]] : i32, !llvm.ptr
// CHECK: %[[COPY_CALL_ARGS_HANDLE:[0-9]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[COPY_CALL_ARGS_HANDLE_PTR:[0-9]+]] = llvm.getelementptr %[[COPY_CALL_ARGS]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY_CALL_ARGS_HANDLE]], %[[COPY_CALL_ARGS_HANDLE_PTR]] : i64, !llvm.ptr
// CHECK: %[[COPY_CALL_STATUS:[0-9]+]] = llvm.call @TVMFFIFunctionCall(%[[COPY_HANDLE]], %[[COPY_ARGUMENTS_SLOT]], %[[COPY_NUM_ARGS]], %[[COPY_CALL_ARGS]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: return
func.func @copy(%destination: !tvm_ffi.tensor, %source: !tvm_ffi.tensor) {
  tvm_ffi.tensor.copy_ %destination, %source : !tvm_ffi.tensor
  return
}
