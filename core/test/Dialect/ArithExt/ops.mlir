//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s | FileCheck %s

// CHECK-LABEL: func.func @and_then_zero_regions
// CHECK: %[[RESULT:.*]] = "arithext.and_then"() : () -> i1
// CHECK: return %[[RESULT]] : i1
func.func @and_then_zero_regions() -> i1 {
  %result = "arithext.and_then"() : () -> i1
  return %result : i1
}

// CHECK-LABEL: func.func @and_then_short_circuit
// CHECK: %[[RESULT:.*]] = "arithext.and_then"() ({
// CHECK-NEXT:   %[[CONTINUE:.*]] = arith.constant true
// CHECK-NEXT:   arithext.and_then.yield %[[CONTINUE]] : i1
// CHECK-NEXT: }, {
// CHECK-NEXT:   %[[STOP:.*]] = arith.constant false
// CHECK-NEXT:   arithext.and_then.yield %[[STOP]] : i1
// CHECK-NEXT: }, {
// CHECK-NEXT:   %[[UNREACHED:.*]] = arith.constant true
// CHECK-NEXT:   arithext.and_then.yield %[[UNREACHED]] : i1
// CHECK-NEXT: }) : () -> i1
// CHECK: return %[[RESULT]] : i1
func.func @and_then_short_circuit() -> i1 {
  %result = "arithext.and_then"() ({
    %continue = arith.constant true
    arithext.and_then.yield %continue : i1
  }, {
    %stop = arith.constant false
    arithext.and_then.yield %stop : i1
  }, {
    %unreached = arith.constant true
    arithext.and_then.yield %unreached : i1
  }) : () -> i1
  return %result : i1
}

// CHECK-LABEL: func.func @and_then_generic
// CHECK: "arithext.and_then"() ({
func.func @and_then_generic() -> i1 {
  %result = "arithext.and_then"() ({
    %success = arith.constant true
    arithext.and_then.yield %success : i1
  }) : () -> i1
  return %result : i1
}
