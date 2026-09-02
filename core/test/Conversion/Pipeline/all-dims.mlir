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
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// The first dispatch constructs the FFI Array for the list of dimensions.
// CHECK: llvm.call @TVMFFIFunctionCall(%[[ARRAY_HANDLE:[0-9]+]], %[[ARRAY_ARGS:[0-9]+]], %[[ARRAY_ARG_COUNT:[0-9]+]], %[[ARRAY_RETURN_SLOT:[0-9]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[ARRAY_HANDLE]])
// CHECK: %[[GETGLOBAL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[ATEN_NAME:[0-9]+]], %[[HANDLE_SLOT:[0-9]+]])
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[ARGS_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[ARG0]], %[[ARGS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[DIMS_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][1]
// CHECK: llvm.store %[[DIMS:[a-zA-Z0-9_]+]], %[[DIMS_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[KEEPDIM_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][2]
// CHECK: llvm.store %[[KEEPDIM:[a-zA-Z0-9_]+]], %[[KEEPDIM_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[RET_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[RET_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[CALL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS_COPY:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]])
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_all_dims(
// CHECK: llvm.call @all_dims(%[[WRAPPER_INPUT:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>

func.func @torch.aten.all.dims(%arg0: !torch.vtensor<[7,4,11,1],f32>) -> !torch.vtensor<[11,1],i1> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %dims = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %result = torch.operator "torch.aten.all.dims"(%arg0, %dims, %false) : (!torch.vtensor<[7,4,11,1],f32>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[11,1],i1>
  return %result : !torch.vtensor<[11,1],i1>
}

tvm_ffi.func @all_dims(%arg0: !torch.vtensor<[7,4,11,1],f32>) -> !torch.vtensor<[11,1],i1> attributes {emit_tvm_ffi_abi} {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %dims = torch.prim.ListConstruct %int1, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %result = torch.operator "torch.aten.all.dims"(%arg0, %dims, %false) : (!torch.vtensor<[7,4,11,1],f32>, !torch.list<int>, !torch.bool) -> !torch.vtensor<[11,1],i1>
  tvm_ffi.return %result : !torch.vtensor<[11,1],i1>
}
