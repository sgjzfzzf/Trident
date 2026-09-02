//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -finalize-tvm-ffi -convert-tvm-ffi-to-llvm -convert-dlpack-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @tensor_metadata(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[INDEX:[a-zA-Z0-9_]+]]: i64)
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[OBJECT:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[PAYLOAD]] : i64 to !llvm.ptr
// CHECK: %[[DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[OBJECT]][24]
// CHECK: %[[DLTENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[DLTENSOR_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[DIM_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DLTENSOR]][2]
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = llvm.sext %[[DIM_BITS]] : i32 to i64
// CHECK: %[[SIZE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[SIZE_TENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[SIZE_PAYLOAD]] : i64 to !llvm.ptr
// CHECK: %[[SIZE_DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SIZE_TENSOR_PTR]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[SIZE_DLTENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[SIZE_DLTENSOR_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[SIZE_STORAGE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[SIZE_DLTENSOR]][4]
// CHECK: %[[SIZE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[SIZE_STORAGE]][%[[INDEX]]]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = llvm.load %[[SIZE_PTR]] : !llvm.ptr -> i64
// CHECK: %[[STRIDE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[STRIDE_TENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[STRIDE_PAYLOAD]] : i64 to !llvm.ptr
// CHECK: %[[STRIDE_DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[STRIDE_TENSOR_PTR]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[STRIDE_DLTENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[STRIDE_DLTENSOR_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[STRIDE_STORAGE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[STRIDE_DLTENSOR]][5]
// CHECK: %[[STRIDE_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[STRIDE_STORAGE]][%[[INDEX]]]
// CHECK: %[[STRIDE:[a-zA-Z0-9_]+]] = llvm.load %[[STRIDE_PTR]] : !llvm.ptr -> i64
// CHECK: %[[OFFSET_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR]][2]
// CHECK: %[[OFFSET_TENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[OFFSET_PAYLOAD]] : i64 to !llvm.ptr
// CHECK: %[[OFFSET_DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[OFFSET_TENSOR_PTR]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[OFFSET_DLTENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[OFFSET_DLTENSOR_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OFFSET_DLTENSOR]][6]
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
// CHECK-SAME: %[[TENSOR2:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK: %[[DTYPE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR2]][2]
// CHECK: %[[DTYPE_TENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[DTYPE_PAYLOAD]] : i64 to !llvm.ptr
// CHECK: %[[DTYPE_DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DTYPE_TENSOR_PTR]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[DTYPE_DLTENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[DTYPE_DLTENSOR_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[DTYPE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DTYPE_DLTENSOR]][3]
// CHECK: %[[CODE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DTYPE]][0]
// CHECK: %[[BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DTYPE]][1]
// CHECK: %[[LANES:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DTYPE]][2]
// CHECK: %[[DEVICE_PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENSOR2]][2]
// CHECK: %[[DEVICE_TENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[DEVICE_PAYLOAD]] : i64 to !llvm.ptr
// CHECK: %[[DEVICE_DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DEVICE_TENSOR_PTR]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK: %[[DEVICE_DLTENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[DEVICE_DLTENSOR_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK: %[[DEVICE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DEVICE_DLTENSOR]][1]
// CHECK: %[[DEVICE_TYPE:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DEVICE]][0]
// CHECK: %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DEVICE]][1]
// CHECK: return %[[CODE]], %[[BITS]], %[[LANES]], %[[DEVICE_TYPE]], %[[DEVICE_INDEX]]
func.func @tensor_dtype_and_device(%tensor: !tvm_ffi.tensor)
    -> (i8, i8, i16, i32, i32) {
  %code, %bits, %lanes = tvm_ffi.tensor.dtype %tensor : !tvm_ffi.tensor
  %device_type, %device_index = tvm_ffi.tensor.device %tensor : !tvm_ffi.tensor
  return %code, %bits, %lanes, %device_type, %device_index
      : i8, i8, i16, i32, i32
}
