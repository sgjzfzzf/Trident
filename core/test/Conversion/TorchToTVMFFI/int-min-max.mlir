//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-torch-to-tvm-ffi | FileCheck %s

// CHECK-LABEL: func.func @int_min_max(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !tvm_ffi.int, %[[RHS:[a-zA-Z0-9_]+]]: !tvm_ffi.int)
// CHECK: %[[MIN_LHS:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[LHS]] : !tvm_ffi.int -> i64
// CHECK: %[[MIN_RHS:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[RHS]] : !tvm_ffi.int -> i64
// CHECK: %[[MIN:[a-zA-Z0-9_]+]] = arith.minsi %[[MIN_LHS]], %[[MIN_RHS]] : i64
// CHECK: %[[MIN_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[MIN]] : i64 -> !tvm_ffi.int
// CHECK: %[[MAX_LHS:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[LHS]] : !tvm_ffi.int -> i64
// CHECK: %[[MAX_RHS:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[RHS]] : !tvm_ffi.int -> i64
// CHECK: %[[MAX:[a-zA-Z0-9_]+]] = arith.maxsi %[[MAX_LHS]], %[[MAX_RHS]] : i64
// CHECK: %[[MAX_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[MAX]] : i64 -> !tvm_ffi.int
// CHECK: return %[[MIN_RESULT]], %[[MAX_RESULT]] : !tvm_ffi.int, !tvm_ffi.int
func.func @int_min_max(%lhs: !torch.int, %rhs: !torch.int)
    -> (!torch.int, !torch.int) {
  %min = torch.prim.min.int %lhs, %rhs
      : !torch.int, !torch.int -> !torch.int
  %max = torch.prim.max.int %lhs, %rhs
      : !torch.int, !torch.int -> !torch.int
  return %min, %max : !torch.int, !torch.int
}
