//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s | FileCheck %s

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
func.func @int_min_max(%lhs: !tvm_ffi.int, %rhs: !tvm_ffi.int) -> (!tvm_ffi.int, !tvm_ffi.int) {
  %min_lhs = tvm_ffi.get %lhs : !tvm_ffi.int -> i64
  %min_rhs = tvm_ffi.get %rhs : !tvm_ffi.int -> i64
  %min = arith.minsi %min_lhs, %min_rhs : i64
  %min_result = tvm_ffi.to %min : i64 -> !tvm_ffi.int
  %max_lhs = tvm_ffi.get %lhs : !tvm_ffi.int -> i64
  %max_rhs = tvm_ffi.get %rhs : !tvm_ffi.int -> i64
  %max = arith.maxsi %max_lhs, %max_rhs : i64
  %max_result = tvm_ffi.to %max : i64 -> !tvm_ffi.int
  return %min_result, %max_result : !tvm_ffi.int, !tvm_ffi.int
}
