//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// The in-place copy keeps the destination as the first tensor argument to the
// runtime copy function and passes the source as the second argument.

// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.runtime.tensor_copy__trident.runtime.tensor_copy_("trident.runtime.tensor_copy_\00")
// CHECK-LABEL: llvm.func @torch.overwrite(
// CHECK-SAME: %[[SOURCE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[DESTINATION:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK: %[[ONE:[0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[DESTINATION_SLOT:[0-9]+]] = llvm.alloca %[[ONE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[SOURCE_SLOT:[0-9]+]] = llvm.alloca %[[ONE]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: llvm.store %[[DESTINATION]], %[[DESTINATION_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: llvm.store %[[SOURCE]], %[[SOURCE_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[ARGUMENTS_COUNT:[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[ARGUMENTS:[0-9]+]] = llvm.alloca %[[ARGUMENTS_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: %[[DESTINATION_VALUE:[0-9]+]] = llvm.load %[[DESTINATION_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[DESTINATION_VALUE]], %[[ARGUMENTS]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[SOURCE_ARGUMENT:[0-9]+]] = llvm.getelementptr %[[ARGUMENTS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK: %[[SOURCE_VALUE:[0-9]+]] = llvm.load %[[SOURCE_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[SOURCE_VALUE]], %[[SOURCE_ARGUMENT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: %[[CALL:[0-9]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE:[0-9]+]], %[[ARGUMENTS]], %[[NARGS:[0-9]+]], %[[CALL_ARGS:[0-9]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: llvm.return

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
