//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -trident-convert-torch-to-scf | FileCheck %s

// CHECK-LABEL: func.func @prim_if(
// CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: !torch.bool,
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !torch.vtensor<[2],f32>,
// CHECK-SAME: %[[RHS:[a-zA-Z0-9_]+]]: !torch.vtensor<[2],f32>)
// CHECK: %[[NATIVE_COND:[a-zA-Z0-9_]+]] = torchext.get %[[COND]] : !torch.bool -> i1
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[NATIVE_COND]] -> (!torch.vtensor<[2],f32>) {
// CHECK-NEXT: scf.yield %[[LHS]] : !torch.vtensor<[2],f32>
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %[[RHS]] : !torch.vtensor<[2],f32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RESULT]] : !torch.vtensor<[2],f32>
func.func @prim_if(%cond: !torch.bool, %lhs: !torch.vtensor<[2],f32>,
                   %rhs: !torch.vtensor<[2],f32>)
    -> !torch.vtensor<[2],f32> {
  %result = torch.prim.If %cond -> (!torch.vtensor<[2],f32>) {
    torch.prim.If.yield %lhs : !torch.vtensor<[2],f32>
  } else {
    torch.prim.If.yield %rhs : !torch.vtensor<[2],f32>
  }
  return %result : !torch.vtensor<[2],f32>
}
