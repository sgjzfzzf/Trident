//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-arith-ext-to-scf | FileCheck %s

// CHECK-LABEL: func.func @and_then
// CHECK: %[[FALSE_VALUE:.*]] = arith.constant false
// CHECK: %[[TRUE_VALUE:.*]] = arith.constant true
// CHECK: %[[RESULT:.*]] = scf.if %[[TRUE_VALUE]] -> (i1) {
// CHECK:   %[[NESTED_FIRST:.*]] = scf.if %[[TRUE_VALUE]] -> (i1) {
// CHECK:     %[[NESTED_SECOND:.*]] = scf.if %[[FALSE_VALUE]] -> (i1) {
// CHECK:       scf.yield %[[TRUE_VALUE]] : i1
// CHECK:     } else {
// CHECK:       scf.yield %[[FALSE_VALUE]] : i1
// CHECK:     }
// CHECK:   scf.yield %[[NESTED_SECOND]] : i1
// CHECK:   } else {
// CHECK:     scf.yield %[[FALSE_VALUE]] : i1
// CHECK:   }
// CHECK:   scf.yield %[[NESTED_FIRST]] : i1
// CHECK: } else {
// CHECK:   scf.yield %[[FALSE_VALUE]] : i1
// CHECK: }
// CHECK: return %[[RESULT]] : i1
func.func @and_then() -> i1 {
  %result = "arithext.and_then"() ({
    %first = arith.constant true
    arithext.and_then.yield %first : i1
  }, {
    %second = arith.constant false
    arithext.and_then.yield %second : i1
  }, {
    %third = arith.constant true
    arithext.and_then.yield %third : i1
  }) : () -> i1
  return %result : i1
}

// CHECK-LABEL: func.func @and_then_empty
// CHECK: %[[EMPTY_RESULT:.*]] = arith.constant true
// CHECK: return %[[EMPTY_RESULT]] : i1
func.func @and_then_empty() -> i1 {
  %result = "arithext.and_then"() : () -> i1
  return %result : i1
}
