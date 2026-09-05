//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: not trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'torchext.trident_kernel_launch'
func.func @torchext_operand_conversion(
    %scalar: !torch.float, %tensor: !torch.vtensor<[4],f32>) {
  %one = arith.constant 1 : i64
  torchext.trident_kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64 threads in (%one, %one, %one)
      args (%tensor : !torch.vtensor<[4],f32> {triton.specialization = #torchext.specialization<kind = !llvm.ptr>}, %scalar : !torch.float {triton.specialization = #torchext.specialization<kind = f64>})
  return
}
