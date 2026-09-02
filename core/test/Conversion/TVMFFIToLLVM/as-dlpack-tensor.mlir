//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @as_dlpack_tensor(
// CHECK-SAME: %[[OBJECT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)> {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[OBJECT]][24]
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[PAYLOAD]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK-NEXT: return %[[TENSOR]] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
func.func @as_dlpack_tensor(%object: !tvm_ffi.object)
    -> !dlpack.dl_tensor {
  %tensor = tvm_ffi.as %object : !tvm_ffi.object -> !dlpack.dl_tensor
  return %tensor : !dlpack.dl_tensor
}
