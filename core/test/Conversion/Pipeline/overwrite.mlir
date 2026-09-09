//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// The in-place copy keeps the destination as the first tensor argument to the
// runtime copy function and passes the source as the second argument.

// CHECK-LABEL: llvm.func @torch.overwrite(
// CHECK-SAME: %[[SOURCE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[DESTINATION:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE:[a-zA-Z0-9_]+]], %[[ARGUMENTS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: llvm.return
// CHECK-LABEL: llvm.func @__tvm_ffi_overwrite(
// CHECK-SAME: %[[WRAPPER_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAPPER_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAPPER_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[WRAPPER_SOURCE:[a-zA-Z0-9_]+]] = llvm.load %[[WRAPPER_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @overwrite(%[[WRAPPER_SOURCE]], %[[WRAPPER_DESTINATION:[a-zA-Z0-9_]+]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> ()

func.func @torch.overwrite(%source: !torch.vtensor<[2,3],f32>,
                           %destination: !torch.tensor<[2,3],f32>) {
  torch.overwrite.tensor.contents %source overwrites %destination
      : !torch.vtensor<[2,3],f32>, !torch.tensor<[2,3],f32>
  return
}

tvm_ffi.func @overwrite(%source: !torch.vtensor<[2,3],f32>,
                         %destination: !torch.tensor<[2,3],f32>)
    attributes {emit_tvm_ffi_abi} {
  torch.overwrite.tensor.contents %source overwrites %destination
      : !torch.vtensor<[2,3],f32>, !torch.tensor<[2,3],f32>
  tvm_ffi.return
}
