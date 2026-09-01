//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @get_float(
// CHECK-SAME:    %[[ARG:[a-zA-Z0-9_]+]]: !torch.float) -> f32 {
// CHECK:         %[[VAL:[a-zA-Z0-9_]+]] = torchext.get %[[ARG]] : !torch.float -> f32
// CHECK-NEXT:    return %[[VAL]] : f32
// CHECK-NEXT:  }
func.func @get_float(%arg0: !torch.float) -> f32 {
  %0 = torchext.get %arg0 : !torch.float -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @get_int(
// CHECK-SAME:    %[[ARG:[a-zA-Z0-9_]+]]: !torch.int) -> i32 {
// CHECK:         %[[VAL:[a-zA-Z0-9_]+]] = torchext.get %[[ARG]] : !torch.int -> i32
// CHECK-NEXT:    return %[[VAL]] : i32
// CHECK-NEXT:  }
func.func @get_int(%arg0: !torch.int) -> i32 {
  %0 = torchext.get %arg0 : !torch.int -> i32
  return %0 : i32
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
