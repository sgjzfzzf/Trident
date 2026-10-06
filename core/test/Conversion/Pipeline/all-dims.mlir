//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// This test verifies that the ATen all.dims call is lowered through the AtenGen FFI dispatch path. In
// particular, the list-valued dimensions operand must be passed as an FFI
// object and the tvm_ffi wrapper must unpack all three arguments.

// CHECK-LABEL: llvm.func @aten.all.dims(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// The first dispatch constructs the FFI Array for the list of dimensions.
// CHECK: llvm.call @TVMFFIFunctionCall(%[[ARRAY_HANDLE:[a-zA-Z0-9_]+]], %[[ARRAY_ARGS:[a-zA-Z0-9_]+]], %[[ARRAY_ARG_COUNT:[a-zA-Z0-9_]+]], %[[ARRAY_RETURN_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// The callee is resolved once at load time, so the dispatch reads the cached
// handle rather than calling TVMFFIFunctionGetGlobal per invocation.
// CHECK: %[[HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.all.dims : !llvm.ptr
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_all_dims(
// CHECK-SAME: %[[WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[WRAPPER_INPUT:[a-zA-Z0-9_]+]] = llvm.load %[[WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAPPER_RET:[a-zA-Z0-9_]+]] = llvm.call @all_dims(%[[WRAPPER_INPUT]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[WRAPPER_RET]], %[[WRAPPER_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr

func.func @aten.all.dims(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %int1 = tvm_ffi.constant.int 1
  %int0 = tvm_ffi.constant.int 0
  %call_dims_handle, %call_dims_lookup = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function, i1
  cf.assert %call_dims_lookup, "lookup failed"
  %dims, %call_dims_status = tvm_ffi.FunctionCall %call_dims_handle(%int1, %int0) : (!tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.array, i1
  cf.assert %call_dims_status, "call failed"
  %false = tvm_ffi.constant.bool false
  %call_result_handle, %call_result_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.all.dims" : !tvm_ffi.function, i1
  cf.assert %call_result_lookup, "lookup failed"
  %result, %call_result_status = tvm_ffi.FunctionCall %call_result_handle(%arg0, %dims, %false) : (!tvm_ffi.tensor, !tvm_ffi.array, !tvm_ffi.bool) -> !tvm_ffi.tensor, i1
  cf.assert %call_result_status, "call failed"
  return %result : !tvm_ffi.tensor
}

tvm_ffi.func @all_dims(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %int1 = tvm_ffi.constant.int 1
  %int0 = tvm_ffi.constant.int 0
  %call_dims_handle, %call_dims_lookup = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function, i1
  cf.assert %call_dims_lookup, "lookup failed"
  %dims, %call_dims_status = tvm_ffi.FunctionCall %call_dims_handle(%int1, %int0) : (!tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.array, i1
  cf.assert %call_dims_status, "call failed"
  %false = tvm_ffi.constant.bool false
  %call_result_handle, %call_result_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.all.dims" : !tvm_ffi.function, i1
  cf.assert %call_result_lookup, "lookup failed"
  %result, %call_result_status = tvm_ffi.FunctionCall %call_result_handle(%arg0, %dims, %false) : (!tvm_ffi.tensor, !tvm_ffi.array, !tvm_ffi.bool) -> !tvm_ffi.tensor, i1
  cf.assert %call_result_status, "call failed"
  tvm_ffi.return %result : !tvm_ffi.tensor
}
