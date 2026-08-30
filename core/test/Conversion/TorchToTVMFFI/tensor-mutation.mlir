//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-torch-to-tvm-ffi | FileCheck %s

// CHECK-LABEL: func.func @copy_to_value(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK-NEXT: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.clone %[[INPUT]] : !tvm_ffi.tensor -> !tvm_ffi.tensor
// CHECK-NEXT: return %[[CLONE]] : !tvm_ffi.tensor
func.func @copy_to_value(%input: !torch.tensor<[2,3],f32>)
    -> !torch.vtensor<[2,3],f32> {
  %copy = torch.copy.to_vtensor %input : !torch.vtensor<[2,3],f32>
  return %copy : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: func.func @overwrite(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[DESTINATION:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) {
// CHECK-NEXT: tvm_ffi.tensor.copy_ %[[DESTINATION]], %[[VALUE]] : !tvm_ffi.tensor
// CHECK-NEXT: return
func.func @overwrite(%value: !torch.vtensor<[2,3],f32>,
                     %destination: !torch.tensor<[2,3],f32>) {
  torch.overwrite.tensor.contents %value overwrites %destination
      : !torch.vtensor<[2,3],f32>, !torch.tensor<[2,3],f32>
  return
}
