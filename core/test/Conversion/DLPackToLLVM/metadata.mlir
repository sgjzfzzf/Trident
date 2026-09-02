//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-dlpack-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @metadata(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, %[[INDEX:[a-zA-Z0-9_]+]]: i64) {
// CHECK: %[[NDIM:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: llvm.sext %[[NDIM]] : i32 to i64
// CHECK: %[[SIZES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][4]
// CHECK: %[[SIZE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SIZES]][%[[INDEX]]]
// CHECK: llvm.load %[[SIZE_PTR]] : !llvm.ptr -> i64
// CHECK: %[[STRIDES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][5]
// CHECK: %[[STRIDE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[STRIDES]][%[[INDEX]]]
// CHECK: llvm.load %[[STRIDE_PTR]] : !llvm.ptr -> i64
// CHECK: llvm.extractvalue %[[TENSOR]][6]
// CHECK: %[[DTYPE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][3]
// CHECK: llvm.extractvalue %[[DTYPE]][0]
// CHECK: llvm.extractvalue %[[DTYPE]][1]
// CHECK: llvm.extractvalue %[[DTYPE]][2]
// CHECK: %[[DEVICE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][1]
// CHECK: llvm.extractvalue %[[DEVICE]][0]
// CHECK: llvm.extractvalue %[[DEVICE]][1]
// CHECK: return
func.func @metadata(%tensor: !dlpack.dl_tensor, %index: i64) {
  %dim = dlpack.tensor.dim %tensor : !dlpack.dl_tensor
  %size = dlpack.tensor.size %tensor[%index] : !dlpack.dl_tensor
  %stride = dlpack.tensor.stride %tensor[%index] : !dlpack.dl_tensor
  %offset = dlpack.tensor.storage_offset %tensor : !dlpack.dl_tensor
  %code, %bits, %lanes = dlpack.tensor.dtype %tensor : !dlpack.dl_tensor
  %device_type, %device_index = dlpack.tensor.device %tensor : !dlpack.dl_tensor
  return
}
