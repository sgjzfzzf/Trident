//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s | FileCheck %s

// CHECK-LABEL: func.func @dlpack_tensor_ops(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !dlpack.dl_tensor, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = dlpack.tensor.dim %[[TENSOR]] : !dlpack.dl_tensor
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = dlpack.tensor.size %[[TENSOR]][%[[INDEX]]] : !dlpack.dl_tensor
// CHECK: %[[STRIDE:[a-zA-Z0-9_]+]] = dlpack.tensor.stride %[[TENSOR]][%[[INDEX]]] : !dlpack.dl_tensor
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = dlpack.tensor.storage_offset %[[TENSOR]] : !dlpack.dl_tensor
// CHECK: %[[CODE:[a-zA-Z0-9_]+]], %[[BITS:[a-zA-Z0-9_]+]], %[[LANES:[a-zA-Z0-9_]+]] = dlpack.tensor.dtype %[[TENSOR]] : !dlpack.dl_tensor
// CHECK: %[[DEVICE_TYPE:[a-zA-Z0-9_]+]], %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = dlpack.tensor.device %[[TENSOR]] : !dlpack.dl_tensor
func.func @dlpack_tensor_ops(%tensor: !dlpack.dl_tensor, %index: i64) {
  %dim = dlpack.tensor.dim %tensor : !dlpack.dl_tensor
  %size = dlpack.tensor.size %tensor[%index] : !dlpack.dl_tensor
  %stride = dlpack.tensor.stride %tensor[%index] : !dlpack.dl_tensor
  %offset = dlpack.tensor.storage_offset %tensor : !dlpack.dl_tensor
  %code, %bits, %lanes = dlpack.tensor.dtype %tensor : !dlpack.dl_tensor
  %device_type, %device_index = dlpack.tensor.device %tensor : !dlpack.dl_tensor
  return
}
