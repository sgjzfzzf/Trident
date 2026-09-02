//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @get_float(
// CHECK-SAME:    %[[ARG:[a-zA-Z0-9_]+]]: !torch.float) -> f64 {
// CHECK:         %[[VAL:[a-zA-Z0-9_]+]] = torchext.get %[[ARG]] : !torch.float -> f64
// CHECK-NEXT:    return %[[VAL]] : f64
// CHECK-NEXT:  }
func.func @get_float(%arg0: !torch.float) -> f64 {
  %0 = torchext.get %arg0 : !torch.float -> f64
  return %0 : f64
}

// -----

// CHECK-LABEL: func.func @get_int(
// CHECK-SAME:    %[[ARG:[a-zA-Z0-9_]+]]: !torch.int) -> i64 {
// CHECK:         %[[VAL:[a-zA-Z0-9_]+]] = torchext.get %[[ARG]] : !torch.int -> i64
// CHECK-NEXT:    return %[[VAL]] : i64
// CHECK-NEXT:  }
func.func @get_int(%arg0: !torch.int) -> i64 {
  %0 = torchext.get %arg0 : !torch.int -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @convert_dtype(
// CHECK-SAME:    %[[DTYPE:[a-zA-Z0-9_]+]]: !torchext.dtype) -> !torch.int {
// CHECK:         %[[SCALAR:[a-zA-Z0-9_]+]] = torchext.convert %[[DTYPE]] : !torchext.dtype -> !torch.int
// CHECK-NEXT:    return %[[SCALAR]] : !torch.int
// CHECK-NEXT:  }
func.func @convert_dtype(%arg0: !torchext.dtype) -> !torch.int {
  %0 = torchext.convert %arg0 : !torchext.dtype -> !torch.int
  return %0 : !torch.int
}

// -----

// CHECK-LABEL: func.func @get_bool(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !torch.bool) -> i1 {
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = torchext.get %[[ARG]] : !torch.bool -> i1
// CHECK-NEXT: return %[[VALUE]] : i1
func.func @get_bool(%arg: !torch.bool) -> i1 {
  %value = torchext.get %arg : !torch.bool -> i1
  return %value : i1
}

// -----

// CHECK-LABEL: func.func @get_tensor_object(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.object {
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = torchext.get %[[ARG]] : !tvm_ffi.tensor -> !tvm_ffi.object
// CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.object
func.func @get_tensor_object(%arg: !tvm_ffi.tensor) -> !tvm_ffi.object {
  %value = torchext.get %arg : !tvm_ffi.tensor -> !tvm_ffi.object
  return %value : !tvm_ffi.object
}
