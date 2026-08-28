//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -canonicalize | FileCheck %s

// Canonicalization must preserve the generalized ATen operator while folding
// ordinary operations from other dialects in the same module.
// CHECK-LABEL: func.func @generalized_operator
// CHECK: %[[CLONE:[a-zA-Z0-9_]+]] = torch.operator "torch.aten.clone"(%arg0,
// CHECK: return %[[CLONE]]
func.func @generalized_operator(%arg0: !torch.vtensor<[3,2],f32>)
    -> !torch.vtensor<[3,2],f32> {
  %format = torch.constant.int 0
  %clone = "torch.aten.clone"(%arg0, %format)
      : (!torch.vtensor<[3,2],f32>, !torch.int)
      -> !torch.vtensor<[3,2],f32>
  return %clone : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL: func.func @other_dialect_folding
// CHECK: %[[THREE:[a-zA-Z0-9_]+]] = arith.constant 3 : i64
// CHECK-NEXT: return %[[THREE]] : i64
func.func @other_dialect_folding() -> i64 {
  %one = arith.constant 1 : i64
  %two = arith.constant 2 : i64
  %sum = arith.addi %one, %two : i64
  return %sum : i64
}
