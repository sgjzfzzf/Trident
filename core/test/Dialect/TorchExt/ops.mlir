//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @kernel_specializations(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[SIZE:[a-zA-Z0-9_]+]]: !tvm_ffi.int) {
// CHECK: torchext.trident_kernel_launch @kernel::@entry
// CHECK-SAME: args(%[[TENSOR]] : !tvm_ffi.tensor #torchext.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %[[SIZE]] : !tvm_ffi.int #torchext.constant_specialization<value = 4 : i64>)
func.func @kernel_specializations(%tensor: !tvm_ffi.tensor,
    %size: !tvm_ffi.int) {
  %one = arith.constant 1 : i64
  torchext.trident_kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64
      threads in (%one, %one, %one)
      args (%tensor : !tvm_ffi.tensor #torchext.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %size : !tvm_ffi.int #torchext.constant_specialization<value = 4 : i64>)
  func.return
}

// -----
