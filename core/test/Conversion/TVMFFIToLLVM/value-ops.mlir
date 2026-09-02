//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @object_type(
// CHECK-SAME: %[[OBJECT:[a-zA-Z0-9_]+]]: !llvm.ptr) {
// CHECK-NEXT: return
func.func @object_type(%object: !tvm_ffi.object) {
  return
}

// CHECK-LABEL: func.func @get_object(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[PAYLOAD]] : i64 to !llvm.ptr
// CHECK-NEXT: return %[[OBJECT]] : !llvm.ptr
func.func @get_object(%tensor: !tvm_ffi.tensor) -> !tvm_ffi.object {
  %object = tvm_ffi.get %tensor : !tvm_ffi.tensor -> !tvm_ffi.object
  return %object : !tvm_ffi.object
}

// CHECK-LABEL: func.func @as_dlpack_tensor(
// CHECK-SAME: %[[OBJECT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)> {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[PAYLOAD]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK-NEXT: return %[[TENSOR]] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
func.func @as_dlpack_tensor(%object: !tvm_ffi.object)
    -> !dlpack.dl_tensor {
  %tensor = tvm_ffi.as %object : !tvm_ffi.object -> !dlpack.dl_tensor
  return %tensor : !dlpack.dl_tensor
}

// CHECK-LABEL: func.func @tensor_literal() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[CREATE_STATUS:[a-zA-Z0-9_]+]] = llvm.call @aoti_torch_create_tensor_from_blob(%[[DATA:[a-zA-Z0-9_]+]], %[[DATA_COUNT:[a-zA-Z0-9_]+]], %[[SHAPE:[a-zA-Z0-9_]+]], %[[STRIDES:[a-zA-Z0-9_]+]], %[[DTYPE:[a-zA-Z0-9_]+]], %[[DEVICE:[a-zA-Z0-9_]+]], %[[REQUIRES_GRAD:[a-zA-Z0-9_]+]], %[[MEMORY_FORMAT:[a-zA-Z0-9_]+]], %[[OUT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[OUT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[EMPTY_STATUS:[a-zA-Z0-9_]+]] = llvm.call @aoti_torch_empty_strided(%[[EMPTY_COUNT:[a-zA-Z0-9_]+]], %[[EMPTY_SHAPE:[a-zA-Z0-9_]+]], %[[EMPTY_STRIDES:[a-zA-Z0-9_]+]], %[[EMPTY_DTYPE:[a-zA-Z0-9_]+]], %[[EMPTY_DEVICE:[a-zA-Z0-9_]+]], %[[EMPTY_REQUIRES_GRAD:[a-zA-Z0-9_]+]], %[[EMPTY_OUT:[a-zA-Z0-9_]+]]) : (i64, !llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
// CHECK: %[[EMPTY_TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_OUT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[COPY_STATUS:[a-zA-Z0-9_]+]] = llvm.call @aoti_torch_copy_(%[[EMPTY_TENSOR]], %[[TENSOR]], %[[NON_BLOCKING:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK: %[[DELETE_STATUS:[a-zA-Z0-9_]+]] = llvm.call @aoti_torch_delete_tensor_object(%[[TENSOR]]) : (!llvm.ptr) -> i32
// CHECK: %[[HANDLE_STATUS:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE:[a-zA-Z0-9_]+]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @tensor_literal() -> !tvm_ffi.tensor {
  %literal = tvm_ffi.tensor.literal dense<[1.0, 2.0]> : tensor<2xf32>
      : !tvm_ffi.tensor
  return %literal : !tvm_ffi.tensor
}

// CHECK-LABEL: func.func @to_native_scalars(
// CHECK-SAME: %[[BOOL_ARG:[a-zA-Z0-9_]+]]: i1, %[[INT_ARG:[a-zA-Z0-9_]+]]: i64, %[[FLOAT_ARG:[a-zA-Z0-9_]+]]: f64)
// CHECK: %[[BOOL_VALUE:[a-zA-Z0-9_]+]] = llvm.zext %[[BOOL_ARG]] : i1 to i64
// CHECK: %[[BOOL_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[INT_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[FLOAT_VALUE:[a-zA-Z0-9_]+]] = llvm.bitcast %[[FLOAT_ARG]] : f64 to i64
// CHECK: %[[FLOAT_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i32)
func.func @to_native_scalars(%bool: i1, %int: i64, %float: f64)
    -> (!tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float) {
  %bool_value = tvm_ffi.to %bool : i1 -> !tvm_ffi.bool
  %int_value = tvm_ffi.to %int : i64 -> !tvm_ffi.int
  %float_value = tvm_ffi.to %float : f64 -> !tvm_ffi.float
  return %bool_value, %int_value, %float_value
      : !tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float
}
