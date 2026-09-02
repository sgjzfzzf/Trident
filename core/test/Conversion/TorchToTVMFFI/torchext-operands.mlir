//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// CHECK-LABEL: func.func @torchext_operand_conversion(
// CHECK-SAME: %[[SCALAR:[a-zA-Z0-9_]+]]: !tvm_ffi.float,
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> f64 {
// CHECK: %[[NATIVE:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[SCALAR]] : !tvm_ffi.float -> f64
// CHECK: torchext.trident_kernel_launch @kernel::@entry
// CHECK-SAME: args(%[[TENSOR]], %[[SCALAR]] : !tvm_ffi.tensor, !tvm_ffi.float)
// CHECK: return %[[NATIVE]] : f64
func.func @torchext_operand_conversion(
    %scalar: !torch.float, %tensor: !torch.vtensor<[4],f32>) -> f64 {
  %one = arith.constant 1 : i64
  %cast = torchext.get %scalar : !torch.float -> f64
  torchext.trident_kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
      args (%tensor, %scalar : !torch.vtensor<[4],f32>, !torch.float)
  return %cast : f64
}
