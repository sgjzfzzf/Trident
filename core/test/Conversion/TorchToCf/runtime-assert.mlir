//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-torch-to-cf | FileCheck %s

// CHECK-LABEL:   func.func @torch.runtime.assert(
// CHECK-SAME:      %[[X:[a-zA-Z0-9_]+]]: !torch.int,
// CHECK-SAME:      %[[Y:[a-zA-Z0-9_]+]]: !torch.int) {
// CHECK:           %[[CMP:[a-zA-Z0-9_]+]] = torch.aten.ne.int %[[X]], %[[Y]] : !torch.int, !torch.int -> !torch.bool
// CHECK-NEXT:      %[[CAST:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[CMP]] : !torch.bool to i1
// CHECK-NEXT:      cf.assert %[[CAST]], "x must not be equal to y"
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @torch.runtime.assert(%arg0: !torch.int, %arg1: !torch.int) {
  %0 = torch.aten.ne.int %arg0, %arg1 : !torch.int, !torch.int -> !torch.bool
  torch.runtime.assert %0, "x must not be equal to y"
  return
}

// CHECK-LABEL:   func.func @torch.runtime.assert_direct(
// CHECK-SAME:      %[[COND:[a-zA-Z0-9_]+]]: !torch.bool) {
// CHECK:           %[[CAST:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[COND]] : !torch.bool to i1
// CHECK-NEXT:      cf.assert %[[CAST]], "direct bool assert"
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @torch.runtime.assert_direct(%arg0: !torch.bool) {
  torch.runtime.assert %arg0, "direct bool assert"
  return
}
