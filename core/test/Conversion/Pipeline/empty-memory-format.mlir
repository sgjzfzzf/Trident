//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that torch.aten.empty.memory_format is lowered via the
// AtenGen FFI dispatch path: "trident.aten.empty.memory_format", called via
// TVMFFIFunctionGetGlobal / TVMFFIFunctionCall / TVMFFIObjectDecRef.

// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.empty.memory_format_trident.aten.empty.memory_format("trident.aten.empty.memory_format\00")
// CHECK-LABEL:   llvm.func @torch.aten.empty.memory_format
// CHECK-SAME: %[[SHAPE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[DTYPE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[DEVICE_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[FUNCTION_NAME:[0-9]+]], %[[HANDLE_SLOT:[0-9]+]])
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[ARGS_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[SHAPE_ARG]], %[[ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[DTYPE_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][1]
// CHECK: llvm.store %[[DTYPE_ARG]], %[[DTYPE_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[LAYOUT_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][2]
// CHECK: llvm.store %[[LAYOUT:[a-zA-Z0-9_]+]], %[[LAYOUT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[DEVICE_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][3]
// CHECK: llvm.store %[[DEVICE_ARG]], %[[DEVICE_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE:[0-9]+]], %[[ARGS_COPY:[0-9]+]], %[[NARGS:[0-9]+]], %[[RET_SLOT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]])
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_empty(
// CHECK: llvm.call @empty(%[[WRAPPER_SHAPE:[a-zA-Z0-9_]+]], %[[WRAPPER_DTYPE:[a-zA-Z0-9_]+]], %[[WRAPPER_DEVICE:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>

// Allocate the args array for 6 operands.
// Store the input operands.
// Build the function name struct {ptr, i64} and call TVMFFIFunctionGetGlobal.
// Load the function handle from the result slot.
// Set up the return value slot and call TVMFFIFunctionCall with the args.
// Release the function handle.
// Load the result TVMFFIAny from the return slot.
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
