//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// This test verifies that torch.aten.all.dims, represented by a generic
// torch.operator, is lowered through the AtenGen FFI dispatch path. In
// particular, the list-valued dimensions operand must be passed as an FFI
// object and the tvm_ffi wrapper must unpack all three arguments.

// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.all.dims_trident.aten.all.dims("trident.aten.all.dims\00")
// CHECK-LABEL: llvm.func @torch.aten.all.dims(
// CHECK-SAME: %[[ARG0:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// The first dispatch constructs the FFI Array for the list of dimensions.
// CHECK: llvm.call @TVMFFIFunctionCall(%[[ARRAY_HANDLE:[0-9]+]], %[[ARRAY_ARGS:[0-9]+]], %[[ARRAY_ARG_COUNT:[0-9]+]], %[[ARRAY_RETURN_SLOT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[ARRAY_HANDLE]])
// CHECK: %[[GETGLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal(%[[ATEN_NAME:[0-9]+]], %[[HANDLE_SLOT:[0-9]+]])
// CHECK: %[[HANDLE:.*]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[ARGS:.*]] = llvm.alloca %[[ARGS_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[ARG0]], %[[ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[DIMS_SLOT:.*]] = llvm.getelementptr %[[ARGS]][1]
// CHECK: llvm.store %[[DIMS:.*]], %[[DIMS_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[KEEPDIM_SLOT:.*]] = llvm.getelementptr %[[ARGS]][2]
// CHECK: llvm.store %[[KEEPDIM:.*]], %[[KEEPDIM_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[RET_SLOT:.*]] = llvm.alloca %[[RET_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[CALL:.*]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS_COPY:.*]], %[[NARGS:.*]], %[[RET_SLOT]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]])
// CHECK: %[[RET:.*]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[RESULT_OBJECT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[DIMS_OBJECT:[0-9]+]])
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_all_dims(
// CHECK-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK: %[[WRAP_ARG0:.*]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[WRAP_ARRAY_NAME:[0-9]+]], %[[WRAP_ARRAY_HANDLE_SLOT:[0-9]+]])
// CHECK: llvm.call @TVMFFIFunctionCall(%[[WRAP_ARRAY_HANDLE:[0-9]+]], %[[WRAP_ARRAY_ARGS:[0-9]+]], %[[WRAP_ARRAY_ARG_COUNT:[0-9]+]], %[[WRAP_ARRAY_RETURN_SLOT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[WRAP_ARRAY_HANDLE]])
// CHECK: %[[WRAP_GETGLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal(%[[WRAP_ATEN_NAME:[0-9]+]], %[[WRAP_HANDLE_SLOT:[0-9]+]])
// CHECK: %[[WRAP_HANDLE:.*]] = llvm.load %[[WRAP_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[WRAP_ARGS:.*]] = llvm.alloca %[[WRAP_ARGS_COUNT:.*]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[WRAP_ARG0]], %[[WRAP_ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[WRAP_DIMS_SLOT:.*]] = llvm.getelementptr %[[WRAP_ARGS]][1]
// CHECK: llvm.store %[[WRAP_DIMS:.*]], %[[WRAP_DIMS_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[WRAP_KEEPDIM_SLOT:.*]] = llvm.getelementptr %[[WRAP_ARGS]][2]
// CHECK: llvm.store %[[WRAP_KEEPDIM:.*]], %[[WRAP_KEEPDIM_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[WRAP_CALL:.*]] = llvm.call @TVMFFIFunctionCall(%[[WRAP_HANDLE]], %[[WRAP_ARGS_COPY:.*]], %[[WRAP_NARGS:.*]], %[[WRAP_RET_SLOT:.*]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[WRAP_HANDLE]])
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[WRAP_RESULT_OBJECT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[WRAP_DIMS_OBJECT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[WRAP_FINAL_OBJECT:[0-9]+]])
// CHECK: llvm.store %[[WRAP_RET:.*]], %arg3

func.func @torch.aten.all.dims(%arg0: !torch.vtensor<[7,4,11,1],f32>) -> !torch.vtensor<[11,1],i1> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %dims = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %result = torch.operator "torch.aten.all.dims"(%arg0, %dims, %false) : (!torch.vtensor<[7,4,11,1],f32>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[11,1],i1>
  return %result : !torch.vtensor<[11,1],i1>
}

tvm_ffi.func @all_dims(%arg0: !torch.vtensor<[7,4,11,1],f32>) -> !torch.vtensor<[11,1],i1> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %dims = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %result = torch.operator "torch.aten.all.dims"(%arg0, %dims, %false) : (!torch.vtensor<[7,4,11,1],f32>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[11,1],i1>
  tvm_ffi.return %result : !torch.vtensor<[11,1],i1>
}
