//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that the ATen transpose view is lowered through the
// AtenGen FFI dispatch path and exposed through the generated TVM FFI wrapper.
//
// NOTE: aten.t is a view op — its result aliases the operand's storage. This
// test only checks that the FFI dispatch path is generated correctly; the
// runtime semantics of the transposed view (strides preserved across the
// DLPack boundary) are exercised end-to-end by test/test_t.py.

// CHECK-LABEL: llvm.func @aten.t(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// The callee is resolved once at load time, so the dispatch reads the cached
// handle rather than calling TVMFFIFunctionGetGlobal per invocation.
// CHECK: %[[HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.t : !llvm.ptr
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS_COPY:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_t(
// CHECK-SAME: %[[WRAP_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAP_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAP_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAP_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[WRAP_ARG:[a-zA-Z0-9_]+]] = llvm.load %[[WRAP_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAP_RET:[a-zA-Z0-9_]+]] = llvm.call @t(%[[WRAP_ARG]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[WRAP_RET]], %[[WRAP_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
func.func @aten.t(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %call_0_handle, %call_0_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.t" : !tvm_ffi.function, i1
  cf.assert %call_0_lookup, "lookup failed"
  %0, %call_0_status = tvm_ffi.FunctionCall %call_0_handle(%arg0) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor, i1
  cf.assert %call_0_status, "call failed"
  return %0 : !tvm_ffi.tensor
}

tvm_ffi.func @t(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %call_0_handle, %call_0_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.t" : !tvm_ffi.function, i1
  cf.assert %call_0_lookup, "lookup failed"
  %0, %call_0_status = tvm_ffi.FunctionCall %call_0_handle(%arg0) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor, i1
  cf.assert %call_0_status, "call failed"
  tvm_ffi.return %0 : !tvm_ffi.tensor
}
