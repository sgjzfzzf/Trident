//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi -convert-scf-to-cf -ownership-deallocation | FileCheck %s

// Torch-to-TVMFFI preserves the literal as a semantic TVMFFI operation. Its
// runtime allocation is deliberately deferred to TVMFFI-to-LLVM.
// CHECK-LABEL: func.func @literal_standalone() -> !tvm_ffi.tensor {
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.literal dense<1.250000e+00> : tensor<2x3xf32> : !tvm_ffi.tensor
// CHECK: return %[[TENSOR]] : !tvm_ffi.tensor
func.func @literal_standalone() -> !torch.vtensor<[2,3],f32> {
  %literal = torch.vtensor.literal(
      dense<1.250000e+00> : tensor<2x3xf32>)
      : !torch.vtensor<[2,3],f32>
  return %literal : !torch.vtensor<[2,3],f32>
}
