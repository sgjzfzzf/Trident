//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// TorchExt operations remain in this IR stage, but all of their operand and
// result types must be legal according to the shared type converter.
// CHECK-LABEL: func.func @torchext_operand_conversion(
// CHECK-SAME: %[[SCALAR:[a-zA-Z0-9_]+]]: !tvm_ffi.float,
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> f32 {
// CHECK: %[[CAST:[a-zA-Z0-9_]+]] = torchext.cast %[[SCALAR]] : !tvm_ffi.float -> f32
// CHECK: torchext.trident_kernel_launch @kernel::@entry
// CHECK-SAME: args(%[[TENSOR]], %[[SCALAR]] : !tvm_ffi.tensor, !tvm_ffi.float)
// CHECK: return %[[CAST]] : f32
func.func @torchext_operand_conversion(
    %scalar: !torch.float, %tensor: !torch.vtensor<[4],f32>) -> f32 {
  %one = arith.constant 1 : i64
  %cast = torchext.cast %scalar : !torch.float -> f32
  torchext.trident_kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) threads in (%one, %one, %one)
      args (%tensor, %scalar : !torch.vtensor<[4],f32>, !torch.float)
  return %cast : f32
}
