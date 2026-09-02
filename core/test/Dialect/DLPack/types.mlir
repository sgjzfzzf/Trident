//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s | FileCheck %s

// CHECK-LABEL: func.func @dl_tensor_type(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !dlpack.dl_tensor) -> !dlpack.dl_tensor {
// CHECK-NEXT: return %[[TENSOR]] : !dlpack.dl_tensor
func.func @dl_tensor_type(%tensor: !dlpack.dl_tensor) -> !dlpack.dl_tensor {
  return %tensor : !dlpack.dl_tensor
}

// CHECK-LABEL: func.func @dl_device_type(
// CHECK-SAME: %[[DEVICE:[a-zA-Z0-9_]+]]: !dlpack.dl_device) -> !dlpack.dl_device {
// CHECK-NEXT: return %[[DEVICE]] : !dlpack.dl_device
func.func @dl_device_type(%device: !dlpack.dl_device) -> !dlpack.dl_device {
  return %device : !dlpack.dl_device
}

// CHECK-LABEL: func.func @dl_data_type(
// CHECK-SAME: %[[DTYPE:[a-zA-Z0-9_]+]]: !dlpack.dl_data_type) -> !dlpack.dl_data_type {
// CHECK-NEXT: return %[[DTYPE]] : !dlpack.dl_data_type
func.func @dl_data_type(%dtype: !dlpack.dl_data_type) -> !dlpack.dl_data_type {
  return %dtype : !dlpack.dl_data_type
}
