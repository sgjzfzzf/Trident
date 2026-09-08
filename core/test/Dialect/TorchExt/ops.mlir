//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @constant_dtype() -> !torchext.dtype {
// CHECK-NEXT: %[[DTYPE:[a-zA-Z0-9_]+]] = torchext.constant.dtype #torchext.float32
// CHECK-NEXT: return %[[DTYPE]] : !torchext.dtype
// CHECK-NEXT: }
func.func @constant_dtype() -> !torchext.dtype {
  %dtype = torchext.constant.dtype #torchext.float32
  return %dtype : !torchext.dtype
}

// -----

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

// CHECK-LABEL: func.func @eq_tuple(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !torch.tuple<int, str>, %[[RHS:[a-zA-Z0-9_]+]]: !torch.tuple<int, str>) -> i1 {
// CHECK: %[[EQUAL:[a-zA-Z0-9_]+]] = torchext.eq %[[LHS]], %[[RHS]] : !torch.tuple<int, str>
// CHECK-NEXT: return %[[EQUAL]] : i1
func.func @eq_tuple(%lhs: !torch.tuple<int, str>,
    %rhs: !torch.tuple<int, str>) -> i1 {
  %equal = torchext.eq %lhs, %rhs : !torch.tuple<int, str>
  return %equal : i1
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

// CHECK-LABEL: func.func @kernel_specializations(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !torch.vtensor<[4],f32>, %[[SIZE:[a-zA-Z0-9_]+]]: !torch.int) {
// CHECK: torchext.trident_kernel_launch @kernel::@entry
// CHECK-SAME: args(%[[TENSOR]] : !torch.vtensor<[4],f32> #torchext.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %[[SIZE]] : !torch.int #torchext.constant_specialization<value = 4 : i64>)
func.func @kernel_specializations(%tensor: !torch.vtensor<[4],f32>,
    %size: !torch.int) {
  %one = arith.constant 1 : i64
  torchext.trident_kernel_launch @kernel::@entry
      blocks in (%one, %one, %one) : i64
      threads in (%one, %one, %one)
      args (%tensor : !torch.vtensor<[4],f32> #torchext.variable_specialization<kind = !llvm.ptr, divisibility = 16>, %size : !torch.int #torchext.constant_specialization<value = 4 : i64>)
  func.return
}

// -----
