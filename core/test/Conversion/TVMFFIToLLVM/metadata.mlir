//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @tensor_metadata(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[DIM_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[DIM_OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[DIM_BITS]] : i64 to !llvm.ptr
// CHECK: %[[DIM_TENSOR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DIM_OBJECT]][24]
// CHECK: %[[DIM_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DIM_TENSOR]][0, 2]
// CHECK: %[[DIM_I32:[a-zA-Z0-9_]+]] = llvm.load %[[DIM_PTR]] : !llvm.ptr -> i32
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = llvm.sext %[[DIM_I32]] : i32 to i64
// CHECK: %[[SIZE_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[SIZE_OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[SIZE_BITS]] : i64 to !llvm.ptr
// CHECK: %[[SIZE_TENSOR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SIZE_OBJECT]][24]
// CHECK: %[[SHAPE_PTR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SIZE_TENSOR]][0, 4]
// CHECK: %[[SHAPE_PTR:[a-zA-Z0-9_]+]] = llvm.load %[[SHAPE_PTR_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[SIZE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SHAPE_PTR]][%[[INDEX]]]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = llvm.load %[[SIZE_PTR]] : !llvm.ptr -> i64
// CHECK: %[[STRIDE_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[STRIDE_OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[STRIDE_BITS]] : i64 to !llvm.ptr
// CHECK: %[[STRIDE_TENSOR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[STRIDE_OBJECT]][24]
// CHECK: %[[STRIDES_PTR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[STRIDE_TENSOR]][0, 5]
// CHECK: %[[STRIDES_PTR:[a-zA-Z0-9_]+]] = llvm.load %[[STRIDES_PTR_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[STRIDE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[STRIDES_PTR]][%[[INDEX]]]
// CHECK: %[[STRIDE:[a-zA-Z0-9_]+]] = llvm.load %[[STRIDE_PTR]] : !llvm.ptr -> i64
// CHECK: %[[OFFSET_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[OFFSET_OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[OFFSET_BITS]] : i64 to !llvm.ptr
// CHECK: %[[OFFSET_TENSOR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[OFFSET_OBJECT]][24]
// CHECK: %[[OFFSET_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[OFFSET_TENSOR]][0, 6]
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = llvm.load %[[OFFSET_PTR]] : !llvm.ptr -> i64
// CHECK: return %[[DIM]], %[[SIZE]], %[[STRIDE]], %[[OFFSET]] : i64, i64, i64, i64
func.func @tensor_metadata(%tensor: !tvm_ffi.tensor, %index: i64)
    -> (i64, i64, i64, i64) {
  %dim = tvm_ffi.tensor.dim %tensor : !tvm_ffi.tensor
  %size = tvm_ffi.tensor.size %tensor[%index] : !tvm_ffi.tensor
  %stride = tvm_ffi.tensor.stride %tensor[%index] : !tvm_ffi.tensor
  %offset = tvm_ffi.tensor.storage_offset %tensor : !tvm_ffi.tensor
  return %dim, %size, %stride, %offset : i64, i64, i64, i64
}

// -----

// CHECK-LABEL: func.func @tensor_dtype_and_device(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK: %[[DTYPE_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[DTYPE_OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[DTYPE_BITS]] : i64 to !llvm.ptr
// CHECK: %[[DTYPE_TENSOR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DTYPE_OBJECT]][24]
// CHECK: %[[DTYPE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DTYPE_TENSOR]][0, 3]
// CHECK: %[[CODE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DTYPE_PTR]][0, 0]
// CHECK: %[[CODE:[a-zA-Z0-9_]+]] = llvm.load %[[CODE_PTR]] : !llvm.ptr -> i8
// CHECK: %[[BITS_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DTYPE_PTR]][0, 1]
// CHECK: %[[BITS:[a-zA-Z0-9_]+]] = llvm.load %[[BITS_PTR]] : !llvm.ptr -> i8
// CHECK: %[[LANES_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DTYPE_PTR]][0, 2]
// CHECK: %[[LANES:[a-zA-Z0-9_]+]] = llvm.load %[[LANES_PTR]] : !llvm.ptr -> i16
// CHECK: %[[DEVICE_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[DEVICE_OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[DEVICE_BITS]] : i64 to !llvm.ptr
// CHECK: %[[DEVICE_TENSOR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DEVICE_OBJECT]][24]
// CHECK: %[[DEVICE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DEVICE_TENSOR]][0, 1]
// CHECK: %[[TYPE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DEVICE_PTR]][0, 0]
// CHECK: %[[DEVICE_TYPE:[a-zA-Z0-9_]+]] = llvm.load %[[TYPE_PTR]] : !llvm.ptr -> i32
// CHECK: %[[INDEX_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DEVICE_PTR]][0, 1]
// CHECK: %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = llvm.load %[[INDEX_PTR]] : !llvm.ptr -> i32
// CHECK: return %[[CODE]], %[[BITS]], %[[LANES]], %[[DEVICE_TYPE]], %[[DEVICE_INDEX]]
func.func @tensor_dtype_and_device(%tensor: !tvm_ffi.tensor)
    -> (i8, i8, i16, i32, i32) {
  %code, %bits, %lanes = tvm_ffi.tensor.dtype %tensor : !tvm_ffi.tensor
  %device_type, %device_index = tvm_ffi.tensor.device %tensor : !tvm_ffi.tensor
  return %code, %bits, %lanes, %device_type, %device_index
      : i8, i8, i16, i32, i32
}
