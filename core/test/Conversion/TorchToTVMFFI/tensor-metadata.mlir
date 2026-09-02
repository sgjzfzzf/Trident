//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-torch-to-tvm-ffi | FileCheck %s

// CHECK-LABEL: func.func @tensor_metadata(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor,
// CHECK-SAME: %[[INDEX:[a-zA-Z0-9_]+]]: i64) -> (i64, i64, i64, i64, i8, i8, i16, i32, i32) {
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.size %[[TENSOR]][%[[INDEX]]] : !tvm_ffi.tensor
// CHECK: %[[STRIDE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.stride %[[TENSOR]][%[[INDEX]]] : !tvm_ffi.tensor
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.storage_offset %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.dim %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[CODE:[a-zA-Z0-9_]+]], %[[BITS:[a-zA-Z0-9_]+]], %[[LANES:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.dtype %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[DEVICE_TYPE:[a-zA-Z0-9_]+]], %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.device %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: return %[[SIZE]], %[[STRIDE]], %[[OFFSET]], %[[DIM]], %[[CODE]], %[[BITS]], %[[LANES]], %[[DEVICE_TYPE]], %[[DEVICE_INDEX]]
func.func @tensor_metadata(
    %tensor: !torch.vtensor<[?,?],f32>, %index: i64)
    -> (i64, i64, i64, i64, i8, i8, i16, i32, i32) {
  %size = torchext.tensor.size %tensor[%index]
      : !torch.vtensor<[?,?],f32>
  %stride = torchext.tensor.stride %tensor[%index]
      : !torch.vtensor<[?,?],f32>
  %offset = torchext.tensor.storage_offset %tensor
      : !torch.vtensor<[?,?],f32>
  %dim = torchext.tensor.dim %tensor : !torch.vtensor<[?,?],f32>
  %code, %bits, %lanes = torchext.tensor.dtype %tensor
      : !torch.vtensor<[?,?],f32>
  %device_type, %device_index = torchext.tensor.device %tensor
      : !torch.vtensor<[?,?],f32>
  return %size, %stride, %offset, %dim, %code, %bits, %lanes,
      %device_type, %device_index : i64, i64, i64, i64, i8, i8, i16, i32, i32
}
