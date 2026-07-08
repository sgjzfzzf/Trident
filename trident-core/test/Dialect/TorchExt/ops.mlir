//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @object_inc_ref
func.func @object_inc_ref(%object: !torch.vtensor<[3,4],f32>) {
  // CHECK: torchext.ObjectIncRef %[[OBJ:.*]] : !torch.vtensor<[3,4],f32>
  torchext.ObjectIncRef %object : !torch.vtensor<[3,4],f32>
  return
}

// -----

// CHECK-LABEL: func.func @object_dec_ref
func.func @object_dec_ref(%object: !torch.list<int>) {
  // CHECK: torchext.ObjectDecRef %[[OBJ:.*]] : !torch.list<int>
  torchext.ObjectDecRef %object : !torch.list<int>
  return
}

// -----

// CHECK-LABEL: func.func @cast_float
func.func @cast_float(%arg0: !torch.float) -> f32 {
  // CHECK: torchext.cast %[[VAL:.*]] : !torch.float -> f32
  %0 = torchext.cast %arg0 : !torch.float -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @cast_int
func.func @cast_int(%arg0: !torch.int) -> i32 {
  // CHECK: torchext.cast %[[VAL:.*]] : !torch.int -> i32
  %0 = torchext.cast %arg0 : !torch.int -> i32
  return %0 : i32
}
