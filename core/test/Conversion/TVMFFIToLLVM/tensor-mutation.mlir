//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @clone(
// CHECK-SAME: %[[CLONE_INPUT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[CLONE_SIZE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[CLONE_INPUTS:[a-zA-Z0-9_]+]] = llvm.alloca %[[CLONE_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[CLONE_INPUT]], %[[CLONE_INPUTS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[CLONE_CALL_INPUTS:[a-zA-Z0-9_]+]] = llvm.alloca %[[CLONE_CALL_SIZE:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[CLONE_CALL_INPUT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[CLONE_CALL_INPUTS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: %[[CLONE_INPUT_VALUE:[a-zA-Z0-9_]+]] = llvm.load %[[CLONE_INPUTS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[CLONE_INPUT_VALUE]], %[[CLONE_CALL_INPUT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[CLONE_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[CLONE_HANDLE_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[CLONE_HANDLE]], %[[CLONE_CALL_INPUTS]], %[[CLONE_NARGS:[a-zA-Z0-9_]+]], %[[CLONE_RESULTS:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[CLONE_RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[CLONE_RESULTS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[CLONE_RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @clone(%input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %copy = tvm_ffi.tensor.clone %input : !tvm_ffi.tensor -> !tvm_ffi.tensor
  return %copy : !tvm_ffi.tensor
}

// CHECK-LABEL: func.func @copy(
// CHECK-SAME: %[[COPY_DESTINATION:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[COPY_SOURCE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK: %[[COPY_SIZE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[COPY_DEST_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[COPY_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[COPY_SOURCE_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[COPY_SIZE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[COPY_DESTINATION]], %[[COPY_DEST_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.store %[[COPY_SOURCE]], %[[COPY_SOURCE_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[COPY_ARGUMENTS_COUNT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[COPY_ARGUMENTS_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[COPY_ARGUMENTS_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[COPY_DESTINATION_ARGUMENT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[COPY_ARGUMENTS_SLOT]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: %[[COPY_DESTINATION_VALUE:[a-zA-Z0-9_]+]] = llvm.load %[[COPY_DEST_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY_DESTINATION_VALUE]], %[[COPY_DESTINATION_ARGUMENT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[COPY_SOURCE_ARGUMENT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[COPY_ARGUMENTS_SLOT]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: %[[COPY_SOURCE_VALUE:[a-zA-Z0-9_]+]] = llvm.load %[[COPY_SOURCE_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY_SOURCE_VALUE]], %[[COPY_SOURCE_ARGUMENT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[COPY_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[COPY_HANDLE_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[COPY_HANDLE]], %[[COPY_ARGUMENTS_SLOT]], %[[COPY_NUM_ARGS:[a-zA-Z0-9_]+]], %[[COPY_RESULTS:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: return
func.func @copy(%destination: !tvm_ffi.tensor, %source: !tvm_ffi.tensor) {
  tvm_ffi.tensor.copy_ %destination, %source : !tvm_ffi.tensor
  return
}
