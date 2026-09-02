//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that torch.aten.t (transpose view) is lowered via the
// AtenGen FFI dispatch path: "trident.aten.t", called via
// TVMFFIFunctionGetGlobal / TVMFFIFunctionCall / TVMFFIObjectDecRef.
//
// NOTE: aten.t is a view op — its result aliases the operand's storage. This
// test only checks that the FFI dispatch path is generated correctly; the
// runtime semantics of the transposed view (strides preserved across the
// DLPack boundary) are exercised end-to-end by test/test_t.py.

// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.t_trident.aten.t("trident.aten.t\00")
// CHECK-LABEL: llvm.func @torch.aten.t(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[GETGLOBAL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[FUNCTION_NAME:[0-9]+]], %[[HANDLE_SLOT:[0-9]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[ARGS_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[ARG0]], %[[ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE:[0-9]+]], %[[ARGS_COPY:[0-9]+]], %[[NARGS:[0-9]+]], %[[RET_SLOT:[0-9]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_t(
// CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK: %[[WRAP_ARG:[a-zA-Z0-9_]+]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAP_GETGLOBAL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[WRAP_FUNCTION_NAME:[0-9]+]], %[[WRAP_HANDLE_SLOT:[0-9]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[WRAP_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[WRAP_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[WRAP_CALL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[WRAP_HANDLE]], %[[WRAP_ARGS:[0-9]+]], %[[WRAP_NARGS:[0-9]+]], %[[WRAP_RET_SLOT:[0-9]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[WRAP_HANDLE]]) : (!llvm.ptr) -> i32
// The local helper is inlined before ref-count insertion, so the returned
// tensor gets exactly one escaping reference at the wrapper boundary.
// CHECK: llvm.store %[[WRAP_RET:[a-zA-Z0-9_]+]], %arg3

// Allocate the args array for 1 operand.
// Store the input operand.
// Build the function name struct {ptr, i64} and call TVMFFIFunctionGetGlobal.
// Load the function handle from the result slot.
// Set up the return value slot and call TVMFFIFunctionCall with the args.
// Release the function handle.
// Load the result TVMFFIAny from the return slot.
func.func @torch.aten.t(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

tvm_ffi.func @t(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  tvm_ffi.return %0 : !torch.vtensor<[3,2],f32>
}
