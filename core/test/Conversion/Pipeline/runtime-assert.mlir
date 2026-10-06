//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// CHECK-LABEL: llvm.func @assert_distinct(
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: i64, %[[Y:[a-zA-Z0-9_]+]]: i64
// CHECK: %[[CMP:[a-zA-Z0-9_]+]] = llvm.icmp "ne" %[[X]], %[[Y]] : i64
// CHECK: llvm.cond_br %[[CMP]],
func.func @assert_distinct(%x: i64, %y: i64) {
  %different = arith.cmpi ne, %x, %y : i64
  cf.assert %different, "x must not be equal to y"
  return
}

// CHECK-LABEL: llvm.func @assert_direct(
// CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1
// CHECK: llvm.cond_br %[[COND]],
func.func @assert_direct(%condition: i1) {
  cf.assert %condition, "direct bool assert"
  return
}
