//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// CHECK-LABEL: llvm.func @conditional(
// CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[LHS:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[RHS:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.select %[[COND]], %[[LHS]], %[[RHS]]
// CHECK: llvm.return %[[RESULT]]
func.func @conditional(%cond: i1, %lhs: !tvm_ffi.tensor, %rhs: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  %result = scf.if %cond -> (!tvm_ffi.tensor) {
    scf.yield %lhs : !tvm_ffi.tensor
  } else {
    scf.yield %rhs : !tvm_ffi.tensor
  }
  return %result : !tvm_ffi.tensor
}
