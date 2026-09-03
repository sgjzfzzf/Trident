//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @get_float(
// CHECK-SAME:    %[[ARG:[a-zA-Z0-9_]+]]: !torch.float) -> f64 {
// CHECK:         %[[VAL:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG]]
// CHECK-NEXT:    return %[[VAL]] : f64
// CHECK-NEXT:  }
func.func @get_float(%arg0: !torch.float) -> f64 {
  %0 = torch_c.to_f64 %arg0
  return %0 : f64
}

// -----

// CHECK-LABEL: func.func @get_int(
// CHECK-SAME:    %[[ARG:[a-zA-Z0-9_]+]]: !torch.int) -> i64 {
// CHECK:         %[[VAL:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG]]
// CHECK-NEXT:    return %[[VAL]] : i64
// CHECK-NEXT:  }
func.func @get_int(%arg0: !torch.int) -> i64 {
  %0 = torch_c.to_i64 %arg0
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
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = torch_c.to_i1 %[[ARG]]
// CHECK-NEXT: return %[[VALUE]] : i1
func.func @get_bool(%arg: !torch.bool) -> i1 {
  %value = torch_c.to_i1 %arg
  return %value : i1
}

// -----
