//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @cast_float(
// CHECK-SAME:    %[[ARG:.*]]: !torch.float) -> f32 {
// CHECK:         %[[VAL:.*]] = torchext.cast %[[ARG]] : !torch.float -> f32
// CHECK-NEXT:    return %[[VAL]] : f32
// CHECK-NEXT:  }
func.func @cast_float(%arg0: !torch.float) -> f32 {
  %0 = torchext.cast %arg0 : !torch.float -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @cast_int(
// CHECK-SAME:    %[[ARG:.*]]: !torch.int) -> i32 {
// CHECK:         %[[VAL:.*]] = torchext.cast %[[ARG]] : !torch.int -> i32
// CHECK-NEXT:    return %[[VAL]] : i32
// CHECK-NEXT:  }
func.func @cast_int(%arg0: !torch.int) -> i32 {
  %0 = torchext.cast %arg0 : !torch.int -> i32
  return %0 : i32
}
