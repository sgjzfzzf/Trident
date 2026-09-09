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

// CHECK-LABEL: llvm.func @torch.aten.all.dims(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// The first dispatch constructs the FFI Array for the list of dimensions.
// CHECK: llvm.call @TVMFFIFunctionCall(%[[ARRAY_HANDLE:[a-zA-Z0-9_]+]], %[[ARRAY_ARGS:[a-zA-Z0-9_]+]], %[[ARRAY_ARG_COUNT:[a-zA-Z0-9_]+]], %[[ARRAY_RETURN_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[GETGLOBAL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[ATEN_NAME:[a-zA-Z0-9_]+]], %[[HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_all_dims(
// CHECK-SAME: %[[WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[WRAPPER_INPUT:[a-zA-Z0-9_]+]] = llvm.load %[[WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAPPER_RET:[a-zA-Z0-9_]+]] = llvm.call @all_dims(%[[WRAPPER_INPUT]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[WRAPPER_RET]], %[[WRAPPER_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr

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
