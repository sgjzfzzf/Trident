//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @object_type(
// CHECK-SAME: %[[OBJECT:[a-zA-Z0-9_]+]]: !llvm.ptr) {
// CHECK-NEXT: return
func.func @object_type(%object: !tvm_ffi.object) {
  return
}
