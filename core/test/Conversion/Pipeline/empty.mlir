//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// This test covers the two empty tensor factories that use the AtenGen FFI
// dispatch path: empty_like and empty.memory_format.

// CHECK-LABEL: llvm.func @torch.aten.empty_like
// CHECK-SAME: %[[EMPTY_LIKE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[EMPTY_LIKE_GETGLOBAL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[EMPTY_LIKE_FUNCTION_NAME:[a-zA-Z0-9_]+]], %[[EMPTY_LIKE_HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[EMPTY_LIKE_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_LIKE_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[EMPTY_LIKE_HANDLE]], %[[EMPTY_LIKE_ARGS:[a-zA-Z0-9_]+]], %[[EMPTY_LIKE_NARGS:[a-zA-Z0-9_]+]], %[[EMPTY_LIKE_RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[EMPTY_LIKE_RET:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_LIKE_RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[EMPTY_LIKE_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_empty_like(
// CHECK-SAME: %[[EMPTY_LIKE_WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[EMPTY_LIKE_WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[EMPTY_LIKE_WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[EMPTY_LIKE_WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[EMPTY_LIKE_WRAPPER_ARG:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_LIKE_WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[EMPTY_LIKE_WRAPPER_RET:[a-zA-Z0-9_]+]] = llvm.call @empty_like(%[[EMPTY_LIKE_WRAPPER_ARG]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[EMPTY_LIKE_WRAPPER_RET]], %[[EMPTY_LIKE_WRAPPER_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
func.func @torch.aten.empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return %0 : !torch.vtensor<[200,200,26],f64>
}

// tvm_ffi.func wrapper: calls the registered ATen wrapper through TVM FFI.
tvm_ffi.func @empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> attributes {emit_tvm_ffi_abi} {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  tvm_ffi.return %0 : !torch.vtensor<[200,200,26],f64>
}

// CHECK-LABEL: llvm.func @torch.aten.empty.memory_format
// CHECK-SAME: %[[EMPTY_SHAPE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[EMPTY_DTYPE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[EMPTY_DEVICE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[EMPTY_FUNCTION_NAME:[a-zA-Z0-9_]+]], %[[EMPTY_HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[EMPTY_HANDLE:[a-zA-Z0-9_]+]], %[[EMPTY_ARGS_COPY:[a-zA-Z0-9_]+]], %[[EMPTY_NARGS:[a-zA-Z0-9_]+]], %[[EMPTY_RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[EMPTY_RET:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[EMPTY_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_empty(
// CHECK-SAME: %[[EMPTY_WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[EMPTY_WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[EMPTY_WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[EMPTY_WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[EMPTY_WRAPPER_SHAPE:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[EMPTY_WRAPPER_RET:[a-zA-Z0-9_]+]] = llvm.call @empty(%[[EMPTY_WRAPPER_SHAPE]], %[[EMPTY_WRAPPER_DEVICE:[a-zA-Z0-9_]+]], %[[EMPTY_WRAPPER_DTYPE:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[EMPTY_WRAPPER_RET]], %[[EMPTY_WRAPPER_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
func.func @torch.aten.empty.memory_format(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  return %0 : !torch.vtensor<[?,?],f64>
}

// tvm_ffi.func wrapper: unpacks shape, device, and dtype from TVM FFI args.
tvm_ffi.func @empty(%shape: !torch.list<int>, %device: !torch.Device, %dtype: !torch.int) -> !torch.tensor attributes {emit_tvm_ffi_abi} {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  tvm_ffi.return %0 : !torch.vtensor<[?,?],f64>
}
