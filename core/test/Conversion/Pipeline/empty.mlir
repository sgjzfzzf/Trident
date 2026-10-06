//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// This test covers the two empty tensor factories that use the AtenGen FFI
// dispatch path: empty_like and empty.memory_format.

// Each callee is resolved once at load time, so the dispatch reads the cached
// handle instead of calling TVMFFIFunctionGetGlobal per invocation.
// CHECK-LABEL: llvm.func @aten.empty_like
// CHECK-SAME: %[[EMPTY_LIKE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK-NOT: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: %[[EMPTY_LIKE_HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.empty_like : !llvm.ptr
// CHECK: %[[EMPTY_LIKE_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_LIKE_HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[EMPTY_LIKE_HANDLE]], %[[EMPTY_LIKE_ARGS:[a-zA-Z0-9_]+]], %[[EMPTY_LIKE_NARGS:[a-zA-Z0-9_]+]], %[[EMPTY_LIKE_RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[EMPTY_LIKE_RET:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_LIKE_RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[EMPTY_LIKE_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_empty_like(
// CHECK-SAME: %[[EMPTY_LIKE_WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[EMPTY_LIKE_WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[EMPTY_LIKE_WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[EMPTY_LIKE_WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[EMPTY_LIKE_WRAPPER_ARG:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_LIKE_WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[EMPTY_LIKE_WRAPPER_RET:[a-zA-Z0-9_]+]] = llvm.call @empty_like(%[[EMPTY_LIKE_WRAPPER_ARG]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[EMPTY_LIKE_WRAPPER_RET]], %[[EMPTY_LIKE_WRAPPER_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
func.func @aten.empty_like(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %none = tvm_ffi.constant.none
  %false = tvm_ffi.constant.bool false
  %call_0_handle, %call_0_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.empty_like" : !tvm_ffi.function, i1
  cf.assert %call_0_lookup, "lookup failed"
  %0, %call_0_status = tvm_ffi.FunctionCall %call_0_handle(%arg0, %none, %none, %none, %false, %none) : (!tvm_ffi.tensor, !tvm_ffi.none, !tvm_ffi.none, !tvm_ffi.none, !tvm_ffi.bool, !tvm_ffi.none) -> !tvm_ffi.tensor, i1
  cf.assert %call_0_status, "call failed"
  return %0 : !tvm_ffi.tensor
}

// tvm_ffi.func wrapper: calls the registered ATen wrapper through TVM FFI.
tvm_ffi.func @empty_like(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %none = tvm_ffi.constant.none
  %false = tvm_ffi.constant.bool false
  %call_0_handle, %call_0_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.empty_like" : !tvm_ffi.function, i1
  cf.assert %call_0_lookup, "lookup failed"
  %0, %call_0_status = tvm_ffi.FunctionCall %call_0_handle(%arg0, %none, %none, %none, %false, %none) : (!tvm_ffi.tensor, !tvm_ffi.none, !tvm_ffi.none, !tvm_ffi.none, !tvm_ffi.bool, !tvm_ffi.none) -> !tvm_ffi.tensor, i1
  cf.assert %call_0_status, "call failed"
  tvm_ffi.return %0 : !tvm_ffi.tensor
}

// CHECK-LABEL: llvm.func @aten.empty.memory_format
// CHECK-SAME: %[[EMPTY_SHAPE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[EMPTY_DTYPE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[EMPTY_DEVICE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK-NOT: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: %[[EMPTY_HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.empty.memory_format : !llvm.ptr
// CHECK: %[[EMPTY_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[EMPTY_HANDLE]], %[[EMPTY_ARGS_COPY:[a-zA-Z0-9_]+]], %[[EMPTY_NARGS:[a-zA-Z0-9_]+]], %[[EMPTY_RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[EMPTY_RET:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[EMPTY_RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_empty(
// CHECK-SAME: %[[EMPTY_WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[EMPTY_WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[EMPTY_WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[EMPTY_WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[EMPTY_WRAPPER_SHAPE:[a-zA-Z0-9_]+]] = llvm.load %[[EMPTY_WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[EMPTY_WRAPPER_RET:[a-zA-Z0-9_]+]] = llvm.call @empty(%[[EMPTY_WRAPPER_SHAPE]], %[[EMPTY_WRAPPER_DEVICE:[a-zA-Z0-9_]+]], %[[EMPTY_WRAPPER_DTYPE:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[EMPTY_WRAPPER_RET]], %[[EMPTY_WRAPPER_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
func.func @aten.empty.memory_format(%shape: !tvm_ffi.array, %dtype: !tvm_ffi.int, %device: !tvm_ffi.device) -> !tvm_ffi.tensor {
  %none = tvm_ffi.constant.none
  %layout = tvm_ffi.constant.int 0
  %call_0_handle, %call_0_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.empty.memory_format" : !tvm_ffi.function, i1
  cf.assert %call_0_lookup, "lookup failed"
  %0, %call_0_status = tvm_ffi.FunctionCall %call_0_handle(%shape, %dtype, %layout, %device, %none, %none) : (!tvm_ffi.array, !tvm_ffi.int, !tvm_ffi.int, !tvm_ffi.device, !tvm_ffi.none, !tvm_ffi.none) -> !tvm_ffi.tensor, i1
  cf.assert %call_0_status, "call failed"
  return %0 : !tvm_ffi.tensor
}

// tvm_ffi.func wrapper: unpacks shape, device, and dtype from TVM FFI args.
tvm_ffi.func @empty(%shape: !tvm_ffi.array, %device: !tvm_ffi.device, %dtype: !tvm_ffi.int) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %none = tvm_ffi.constant.none
  %layout = tvm_ffi.constant.int 0
  %call_0_handle, %call_0_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.empty.memory_format" : !tvm_ffi.function, i1
  cf.assert %call_0_lookup, "lookup failed"
  %0, %call_0_status = tvm_ffi.FunctionCall %call_0_handle(%shape, %dtype, %layout, %device, %none, %none) : (!tvm_ffi.array, !tvm_ffi.int, !tvm_ffi.int, !tvm_ffi.device, !tvm_ffi.none, !tvm_ffi.none) -> !tvm_ffi.tensor, i1
  cf.assert %call_0_status, "call failed"
  tvm_ffi.return %0 : !tvm_ffi.tensor
}
