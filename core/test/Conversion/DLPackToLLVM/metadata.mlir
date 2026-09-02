//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-dlpack-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @metadata(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, %[[INDEX:[a-zA-Z0-9_]+]]: i64) {
// CHECK: %[[NDIM:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[NDIM64:[a-zA-Z0-9_]+]] = llvm.sext %[[NDIM]] : i32 to i64
// CHECK: %[[SIZES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][4] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[SIZE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SIZES]][%[[INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK: llvm.load %[[SIZE_PTR]] : !llvm.ptr -> i64
// CHECK: %[[STRIDES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][5] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[STRIDE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[STRIDES]][%[[INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK: llvm.load %[[STRIDE_PTR]] : !llvm.ptr -> i64
// CHECK: %[[BYTE_OFFSET:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][6] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[DTYPE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][3] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[CODE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DTYPE]][0] : !llvm.struct<(i8, i8, i16)>
// CHECK: %[[BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DTYPE]][1] : !llvm.struct<(i8, i8, i16)>
// CHECK: %[[LANES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DTYPE]][2] : !llvm.struct<(i8, i8, i16)>
// CHECK: %[[DEVICE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][1] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[DEVICE_TYPE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DEVICE]][0] : !llvm.struct<(i32, i32)>
// CHECK: %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DEVICE]][1] : !llvm.struct<(i32, i32)>
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
