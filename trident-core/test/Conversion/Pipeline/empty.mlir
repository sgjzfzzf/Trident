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

// CHECK-DAG: llvm.func @TVMFFIObjectIncRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32

// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.empty.memory_format_trident.aten.empty.memory_format("trident.aten.empty.memory_format\00")

// CHECK-LABEL: llvm.func @torch.aten.empty.memory_format(
// CHECK-SAME:    %[[SHAPE:.*]]: !llvm.struct<(i32, i32, i64)>, %[[DTYPE:.*]]: !llvm.struct<(i32, i32, i64)>, %[[DEVICE:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// Allocate the args array for 6 operands.
// CHECK:         %[[ARRAY:.*]] = llvm.alloca %[[CNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// Store the input operands.
// CHECK:         llvm.store %[[SHAPE]], %[[ARRAY]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.store %[[DTYPE]], {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.store %[[DEVICE]], {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// Build the function name struct {ptr, i64} and call TVMFFIFunctionGetGlobal.
// CHECK:         %[[GETGLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal(%[[NAME_SLOT:.*]], %[[HANDLE_SLOT:.*]]) : (!llvm.ptr, !llvm.ptr) -> i32
// Load the function handle from the result slot.
// CHECK:         %[[HANDLE:.*]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// Set up the return value slot and call TVMFFIFunctionCall with the args.
// CHECK:         %[[FUNCCALL:.*]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS_COPY:.*]], %[[NARGS:.*]], %[[RET_SLOT:.*]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// Release the function handle.
// CHECK:         %[[DECREF_HANDLE:.*]] = llvm.call @TVMFFIObjectDecRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
// Load the result TVMFFIAny from the return slot.
// CHECK:         %[[RETLOAD:.*]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// IncRef + DecRef the returned object for correct ref-counting.
// CHECK:         %[[INC:.*]] = llvm.call @TVMFFIObjectIncRef(%[[OBJ_PTR:.*]]) : (!llvm.ptr) -> i32
// CHECK:         %[[DECREF_OBJ:.*]] = llvm.call @TVMFFIObjectDecRef(%[[OBJ_PTR2:.*]]) : (!llvm.ptr) -> i32
// CHECK:         llvm.return %[[RETLOAD]] : !llvm.struct<(i32, i32, i64)>
func.func @torch.aten.empty.memory_format(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  return %0 : !torch.vtensor<[?,?],f64>
}

// tvm_ffi.func wrapper: unpacks shape, device, and dtype from TVM FFI args.
// CHECK-LABEL: llvm.func @__tvm_ffi_empty(
// CHECK-SAME:    %arg0: !llvm.ptr, %[[ARGS:.*]]: !llvm.ptr, %arg2: i32, %[[RET:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[SHAPE_LOAD:.*]] = llvm.load %[[ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[DEV_GEP:.*]] = llvm.getelementptr %[[ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[DEV_LOAD:.*]] = llvm.load %[[DEV_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[DTYPE_GEP:.*]] = llvm.getelementptr %[[ARGS]][2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[DTYPE_LOAD:.*]] = llvm.load %[[DTYPE_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.br ^bb1
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.aten.empty.memory_format(%[[SHAPE_LOAD]], %[[DTYPE_LOAD]], %[[DEV_LOAD]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.store %[[CALLEE_RET]], %[[RET]]
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @empty(%shape: !torch.list<int>, %device: !torch.Device, %dtype: !torch.int) -> !torch.tensor {
  %0 = func.call @torch.aten.empty.memory_format(%shape, %dtype, %device) : (!torch.list<int>, !torch.int, !torch.Device) -> !torch.vtensor<[?,?],f64>
  tvm_ffi.return %0 : !torch.vtensor<[?,?],f64>
}
