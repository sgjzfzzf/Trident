//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// Existing semantic inspection operations keep their operation and result
// types while the remaining Torch operands are converted by the common
// conversion adaptor. Every inspection result remains live through the
// returned guard.
// CHECK-LABEL: tvm_ffi.func @guard_operand_conversion(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor,
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.array,
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !tvm_ffi.bool,
// CHECK-SAME: %[[RHS:[a-zA-Z0-9_]+]]: !tvm_ffi.bool) {
// CHECK-NOT: !torch
// CHECK: %[[EXPECTED_DEVICE_INDEX:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[EXPECTED_DEVICE_TYPE:[a-zA-Z0-9_]+]] = arith.constant 2 : i32
// CHECK: %[[EXPECTED_LANES:[a-zA-Z0-9_]+]] = arith.constant 1 : i16
// CHECK: %[[EXPECTED_BITS:[a-zA-Z0-9_]+]] = arith.constant 32 : i8
// CHECK: %[[EXPECTED_CODE:[a-zA-Z0-9_]+]] = arith.constant 2 : i8
// CHECK: %[[EXPECTED_STRIDE:[a-zA-Z0-9_]+]] = arith.constant 3 : i64
// CHECK: %[[EXPECTED_DIM_SIZE_LENGTH:[a-zA-Z0-9_]+]] = arith.constant 2 : i64
// CHECK: %[[INDEX_AND_EXPECTED_OFFSET:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.dim %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.size %[[TENSOR]][%[[INDEX_AND_EXPECTED_OFFSET]]] : !tvm_ffi.tensor
// CHECK: %[[STRIDE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.stride %[[TENSOR]][%[[INDEX_AND_EXPECTED_OFFSET]]] : !tvm_ffi.tensor
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.storage_offset %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[DTYPE_CODE:[a-zA-Z0-9_]+]], %[[DTYPE_BITS:[a-zA-Z0-9_]+]], %[[DTYPE_LANES:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.dtype %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[DEVICE_TYPE:[a-zA-Z0-9_]+]], %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.device %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[LENGTH:[a-zA-Z0-9_]+]] = tvm_ffi.array.length %[[ARRAY]] : !tvm_ffi.array
// CHECK: %[[VALUES_EQUAL:[a-zA-Z0-9_]+]] = tvm_ffi.eq %[[LHS]], %[[RHS]] : !tvm_ffi.bool, !tvm_ffi.bool
// CHECK: %[[DIM_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DIM]], %[[EXPECTED_DIM_SIZE_LENGTH]] : i64
// CHECK: %[[SIZE_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[SIZE]], %[[EXPECTED_DIM_SIZE_LENGTH]] : i64
// CHECK: %[[STRIDE_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[STRIDE]], %[[EXPECTED_STRIDE]] : i64
// CHECK: %[[OFFSET_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[OFFSET]], %[[INDEX_AND_EXPECTED_OFFSET]] : i64
// CHECK: %[[CODE_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DTYPE_CODE]], %[[EXPECTED_CODE]] : i8
// CHECK: %[[BITS_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DTYPE_BITS]], %[[EXPECTED_BITS]] : i8
// CHECK: %[[LANES_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DTYPE_LANES]], %[[EXPECTED_LANES]] : i16
// CHECK: %[[DEVICE_TYPE_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DEVICE_TYPE]], %[[EXPECTED_DEVICE_TYPE]] : i32
// CHECK: %[[DEVICE_INDEX_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DEVICE_INDEX]], %[[EXPECTED_DEVICE_INDEX]] : i32
// CHECK: %[[LENGTH_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[LENGTH]], %[[EXPECTED_DIM_SIZE_LENGTH]] : i64
// CHECK: %[[SHAPE_OK:[a-zA-Z0-9_]+]] = arith.andi %[[DIM_OK]], %[[SIZE_OK]] : i1
// CHECK: %[[LAYOUT_OK:[a-zA-Z0-9_]+]] = arith.andi %[[STRIDE_OK]], %[[OFFSET_OK]] : i1
// CHECK: %[[DTYPE_HEAD_OK:[a-zA-Z0-9_]+]] = arith.andi %[[CODE_OK]], %[[BITS_OK]] : i1
// CHECK: %[[DTYPE_OK:[a-zA-Z0-9_]+]] = arith.andi %[[DTYPE_HEAD_OK]], %[[LANES_OK]] : i1
// CHECK: %[[DEVICE_OK:[a-zA-Z0-9_]+]] = arith.andi %[[DEVICE_TYPE_OK]], %[[DEVICE_INDEX_OK]] : i1
// CHECK: %[[METADATA_HEAD_OK:[a-zA-Z0-9_]+]] = arith.andi %[[SHAPE_OK]], %[[LAYOUT_OK]] : i1
// CHECK: %[[METADATA_TAIL_OK:[a-zA-Z0-9_]+]] = arith.andi %[[DTYPE_OK]], %[[DEVICE_OK]] : i1
// CHECK: %[[METADATA_OK:[a-zA-Z0-9_]+]] = arith.andi %[[METADATA_HEAD_OK]], %[[METADATA_TAIL_OK]] : i1
// CHECK: %[[CONTAINER_OK:[a-zA-Z0-9_]+]] = arith.andi %[[LENGTH_OK]], %[[VALUES_EQUAL]] : i1
// CHECK: %[[GUARD_OK:[a-zA-Z0-9_]+]] = arith.andi %[[METADATA_OK]], %[[CONTAINER_OK]] : i1
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value}> : () -> !tvm_ffi.none
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.cast %[[NONE]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK: %[[EXCEPTION:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK: %[[ERROR:[a-zA-Z0-9_]+]] = tvm_ffi.cast %[[EXCEPTION]] : !tvm_ffi.exception -> !tvm_ffi.any
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = arith.select %[[GUARD_OK]], %[[SUCCESS]], %[[ERROR]] : !tvm_ffi.any
// CHECK: tvm_ffi.return %[[RESULT]] : !tvm_ffi.any
tvm_ffi.func @guard_operand_conversion(
    %tensor: !tvm_ffi.tensor,
    %array: !torch.list<int>,
    %lhs: !torch.bool,
    %rhs: !torch.bool) {
  %index = arith.constant 0 : i64
  %dim = tvm_ffi.tensor.dim %tensor : !tvm_ffi.tensor
  %size = tvm_ffi.tensor.size %tensor[%index]
      : !tvm_ffi.tensor
  %stride = tvm_ffi.tensor.stride %tensor[%index]
      : !tvm_ffi.tensor
  %offset = tvm_ffi.tensor.storage_offset %tensor
      : !tvm_ffi.tensor
  %dtype_code, %dtype_bits, %dtype_lanes = tvm_ffi.tensor.dtype %tensor
      : !tvm_ffi.tensor
  %device_type, %device_index = tvm_ffi.tensor.device %tensor
      : !tvm_ffi.tensor
  %length = tvm_ffi.array.length %array : !torch.list<int>
  %values_equal = tvm_ffi.eq %lhs, %rhs : !torch.bool, !torch.bool

  %expected_dim = arith.constant 2 : i64
  %expected_size = arith.constant 2 : i64
  %expected_stride = arith.constant 3 : i64
  %expected_offset = arith.constant 0 : i64
  %expected_code = arith.constant 2 : i8
  %expected_bits = arith.constant 32 : i8
  %expected_lanes = arith.constant 1 : i16
  %expected_device_type = arith.constant 2 : i32
  %expected_device_index = arith.constant 0 : i32
  %expected_length = arith.constant 2 : i64
  %dim_ok = arith.cmpi eq, %dim, %expected_dim : i64
  %size_ok = arith.cmpi eq, %size, %expected_size : i64
  %stride_ok = arith.cmpi eq, %stride, %expected_stride : i64
  %offset_ok = arith.cmpi eq, %offset, %expected_offset : i64
  %code_ok = arith.cmpi eq, %dtype_code, %expected_code : i8
  %bits_ok = arith.cmpi eq, %dtype_bits, %expected_bits : i8
  %lanes_ok = arith.cmpi eq, %dtype_lanes, %expected_lanes : i16
  %device_type_ok = arith.cmpi eq, %device_type, %expected_device_type : i32
  %device_index_ok = arith.cmpi eq, %device_index, %expected_device_index : i32
  %length_ok = arith.cmpi eq, %length, %expected_length : i64
  %tensor_shape_ok = arith.andi %dim_ok, %size_ok : i1
  %tensor_layout_ok = arith.andi %stride_ok, %offset_ok : i1
  %tensor_dtype_head_ok = arith.andi %code_ok, %bits_ok : i1
  %tensor_dtype_ok = arith.andi %tensor_dtype_head_ok, %lanes_ok : i1
  %tensor_device_ok = arith.andi %device_type_ok, %device_index_ok : i1
  %tensor_metadata_head_ok = arith.andi %tensor_shape_ok, %tensor_layout_ok : i1
  %tensor_metadata_tail_ok = arith.andi %tensor_dtype_ok, %tensor_device_ok : i1
  %tensor_metadata_ok = arith.andi %tensor_metadata_head_ok,
      %tensor_metadata_tail_ok : i1
  %container_ok = arith.andi %length_ok, %values_equal : i1
  %guard_ok = arith.andi %tensor_metadata_ok, %container_ok : i1

  %none = "tvm_ffi.constant"() <{value}> : () -> !tvm_ffi.none
  %success = tvm_ffi.cast %none : !tvm_ffi.none -> !tvm_ffi.any
  %exception = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  %error = tvm_ffi.cast %exception : !tvm_ffi.exception -> !tvm_ffi.any
  %result = arith.select %guard_ok, %success, %error : !tvm_ffi.any
  tvm_ffi.return %result : !tvm_ffi.any
}
