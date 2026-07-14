//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --eliminate-ref-counter -split-input-file | FileCheck %s

// CHECK-LABEL: @single_pair
// CHECK-NOT: torchext.Object
func.func @single_pair(%arg: !torch.vtensor<[2,3],f32>) {
  torchext.ObjectIncRef %arg : !torch.vtensor<[2,3],f32>
  torchext.ObjectDecRef %arg : !torch.vtensor<[2,3],f32>
  return
}

// -----

// Match independently by value, even when operations are interleaved.
// CHECK-LABEL: @interleaved
// CHECK-NOT: torchext.Object
func.func @interleaved(%lhs: !torch.vtensor<[2,3],f32>,
                       %rhs: !torch.vtensor<[2,3],f32>) {
  torchext.ObjectIncRef %lhs : !torch.vtensor<[2,3],f32>
  torchext.ObjectIncRef %rhs : !torch.vtensor<[2,3],f32>
  torchext.ObjectDecRef %lhs : !torch.vtensor<[2,3],f32>
  torchext.ObjectDecRef %rhs : !torch.vtensor<[2,3],f32>
  return
}

// -----

// A DecRef cannot be paired with a later IncRef.
// CHECK-LABEL: @wrong_order
// CHECK: torchext.ObjectDecRef %[[ARG:.*]] : !torch.vtensor<[2,3],f32>
// CHECK-NEXT: torchext.ObjectIncRef %[[ARG]] : !torch.vtensor<[2,3],f32>
func.func @wrong_order(%arg: !torch.vtensor<[2,3],f32>) {
  torchext.ObjectDecRef %arg : !torch.vtensor<[2,3],f32>
  torchext.ObjectIncRef %arg : !torch.vtensor<[2,3],f32>
  return
}

// -----

// Eliminate as many pairs as possible and preserve unmatched operations.
// CHECK-LABEL: @unbalanced
// CHECK: torchext.ObjectIncRef %[[ARG:.*]] : !torch.vtensor<[2,3],f32>
// CHECK-NOT: torchext.ObjectDecRef
func.func @unbalanced(%arg: !torch.vtensor<[2,3],f32>) {
  torchext.ObjectIncRef %arg : !torch.vtensor<[2,3],f32>
  torchext.ObjectIncRef %arg : !torch.vtensor<[2,3],f32>
  torchext.ObjectDecRef %arg : !torch.vtensor<[2,3],f32>
  return
}

// -----

// Pairs do not cross block boundaries.
// CHECK-LABEL: @different_blocks
// CHECK: torchext.ObjectIncRef %[[ARG:.*]] : !torch.vtensor<[2,3],f32>
// CHECK: ^bb1:
// CHECK-NEXT: torchext.ObjectDecRef %[[ARG]] : !torch.vtensor<[2,3],f32>
func.func @different_blocks(%arg: !torch.vtensor<[2,3],f32>) {
  torchext.ObjectIncRef %arg : !torch.vtensor<[2,3],f32>
  cf.br ^bb1
^bb1:
  torchext.ObjectDecRef %arg : !torch.vtensor<[2,3],f32>
  return
}
