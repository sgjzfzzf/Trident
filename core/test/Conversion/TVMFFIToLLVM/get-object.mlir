//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @get_object(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[PAYLOAD]] : i64 to !llvm.ptr
// CHECK-NEXT: return %[[OBJECT]] : !llvm.ptr
func.func @get_object(%tensor: !tvm_ffi.tensor) -> !tvm_ffi.object {
  %object = tvm_ffi.get %tensor : !tvm_ffi.tensor -> !tvm_ffi.object
  return %object : !tvm_ffi.object
}
