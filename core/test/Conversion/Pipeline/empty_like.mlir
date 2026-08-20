//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that torch.aten.empty_like is lowered via the AtenGen
// FFI dispatch path: "trident.aten.empty_like", called via
// TVMFFIFunctionGetGlobal / TVMFFIFunctionCall / TVMFFIObjectDecRef.

// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.empty_like_trident.aten.empty_like("trident.aten.empty_like\00")
// CHECK-LABEL:   llvm.func @torch.aten.empty_like
// CHECK-SAME: %[[ARG0:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[GETGLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal(%[[FUNCTION_NAME:[0-9]+]], %[[HANDLE_SLOT:[0-9]+]])
// CHECK: %[[HANDLE:.*]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[ARGS:.*]] = llvm.alloca %[[ARGS_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[ARG0]], %[[ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[RET_SLOT:.*]] = llvm.alloca %[[RET_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[CALL:.*]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS_COPY:.*]], %[[NARGS:.*]], %[[RET_SLOT]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]])
// CHECK: %[[RET:.*]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[RESULT_OBJECT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[INPUT_OBJECT:[0-9]+]])
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_empty_like(
// CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK: %[[WRAP_ARG:.*]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAP_GETGLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal(%[[WRAP_FUNCTION_NAME:[0-9]+]], %[[WRAP_HANDLE_SLOT:[0-9]+]])
// CHECK: %[[WRAP_HANDLE:.*]] = llvm.load %[[WRAP_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[WRAP_ARGS:.*]] = llvm.alloca %[[WRAP_ARGS_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[WRAP_RET_SLOT:.*]] = llvm.alloca %[[WRAP_RET_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[WRAP_CALL:.*]] = llvm.call @TVMFFIFunctionCall(%[[WRAP_HANDLE]], %[[WRAP_ARGS_COPY:.*]], %[[WRAP_NARGS:.*]], %[[WRAP_RET_SLOT]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[WRAP_HANDLE]])
// CHECK: %[[WRAP_RET:.*]] = llvm.load %[[WRAP_RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[WRAP_RESULT_OBJECT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[WRAP_INPUT_OBJECT:[0-9]+]])
// CHECK: llvm.store %[[WRAP_RET]], %arg3


// Allocate the args array for 6 operands.
// Store the input operand into the args array.
// Build the function name struct {ptr, i64} and call TVMFFIFunctionGetGlobal.
// Load the function handle from the result slot.
// Set up the return value slot and call TVMFFIFunctionCall with the args.
// Release the function handle.
// Load the result TVMFFIAny from the return slot.
func.func @torch.aten.empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return %0 : !torch.vtensor<[200,200,26],f64>
}

// tvm_ffi.func wrapper: calls the registered ATen wrapper through TVM FFI.
tvm_ffi.func @empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  tvm_ffi.return %0 : !torch.vtensor<[200,200,26],f64>
}
