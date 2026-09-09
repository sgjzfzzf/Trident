//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops | FileCheck %s

// CHECK-LABEL: func.func @add_int(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int)
// CHECK: %[[ADD_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[ADD_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[ADD_RESULT:[a-zA-Z0-9_]+]] = arith.addi %[[ADD_A]], %[[ADD_B]] : i64
// CHECK: %[[ADD_INT:[a-zA-Z0-9_]+]] = torch_c.from_i64 %[[ADD_RESULT]]
func.func @add_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %result = torch.aten.add.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @floordiv_int(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int)
// CHECK: %[[FLOOR_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[FLOOR_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[FLOOR_RESULT:[a-zA-Z0-9_]+]] = arith.floordivsi %[[FLOOR_A]], %[[FLOOR_B]] : i64
// CHECK: torch_c.from_i64 %[[FLOOR_RESULT]]
func.func @floordiv_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %result = torch.aten.floordiv.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @int_bool(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.bool)
// CHECK: %[[BOOL_NATIVE:[a-zA-Z0-9_]+]] = torch_c.to_i1 %[[ARG0]]
// CHECK: %[[BOOL_INT:[a-zA-Z0-9_]+]] = arith.extui %[[BOOL_NATIVE]] : i1 to i64
// CHECK: torch_c.from_i64 %[[BOOL_INT]]
func.func @int_bool(%arg0: !torch.bool) -> !torch.int {
  %result = torch.aten.Int.bool %arg0 : !torch.bool -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @eq_int(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int)
// CHECK: %[[EQ_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[EQ_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[EQ_RESULT:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[EQ_A]], %[[EQ_B]] : i64
// CHECK: torch_c.from_i1 %[[EQ_RESULT]]
func.func @eq_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %result = torch.aten.eq.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.bool
  return %result : !torch.bool
}

// CHECK-LABEL: func.func @le_int(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int)
// CHECK: %[[LE_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[LE_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[LE_RESULT:[a-zA-Z0-9_]+]] = arith.cmpi sle, %[[LE_A]], %[[LE_B]] : i64
// CHECK: torch_c.from_i1 %[[LE_RESULT]]
func.func @le_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.bool {
  %result = torch.aten.le.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.bool
  return %result : !torch.bool
}

// CHECK-LABEL: func.func @mul_int(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int)
// CHECK: %[[MUL_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[MUL_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[MUL_RESULT:[a-zA-Z0-9_]+]] = arith.muli %[[MUL_A]], %[[MUL_B]] : i64
// CHECK: %[[MUL_INT:[a-zA-Z0-9_]+]] = torch_c.from_i64 %[[MUL_RESULT]]
func.func @mul_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %result = torch.aten.mul.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @size_int(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.vtensor<[?,?],f32>, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int)
// CHECK: %[[DIM_NATIVE:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = torchext.tensor.size %[[ARG0]][%[[DIM_NATIVE]]] : !torch.vtensor<[?,?],f32>
// CHECK: torch_c.from_i64 %[[SIZE]]
func.func @size_int(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.int)
    -> !torch.int {
  %result = torch.aten.size.int %arg0, %arg1
      : !torch.vtensor<[?,?],f32>, !torch.int -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @sub_int(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int)
// CHECK: %[[SUB_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[SUB_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[SUB_RESULT:[a-zA-Z0-9_]+]] = arith.subi %[[SUB_A]], %[[SUB_B]] : i64
// CHECK: torch_c.from_i64 %[[SUB_RESULT]]
func.func @sub_int(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %result = torch.aten.sub.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %result : !torch.int
}

// CHECK-LABEL: func.func @float_ops(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.float, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.float)
// CHECK: %[[ADD_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[ADD_FLOAT_B:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG1]]
// CHECK: %[[ADD_FLOAT:[a-zA-Z0-9_]+]] = arith.addf %[[ADD_FLOAT_A]], %[[ADD_FLOAT_B]] : f64
// CHECK: %[[ADD_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_f64 %[[ADD_FLOAT]]
// CHECK: %[[DIV_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[DIV_FLOAT_B:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG1]]
// CHECK: %[[DIV_FLOAT:[a-zA-Z0-9_]+]] = arith.divf %[[DIV_FLOAT_A]], %[[DIV_FLOAT_B]] : f64
// CHECK: %[[DIV_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_f64 %[[DIV_FLOAT]]
// CHECK: %[[MUL_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[MUL_FLOAT_B:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG1]]
// CHECK: %[[MUL_FLOAT:[a-zA-Z0-9_]+]] = arith.mulf %[[MUL_FLOAT_A]], %[[MUL_FLOAT_B]] : f64
// CHECK: %[[MUL_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_f64 %[[MUL_FLOAT]]
// CHECK: %[[NEG_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[NEG_FLOAT:[a-zA-Z0-9_]+]] = arith.negf %[[NEG_FLOAT_A]] : f64
// CHECK: %[[NEG_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_f64 %[[NEG_FLOAT]]
// CHECK: %[[SUB_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[SUB_FLOAT_B:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG1]]
// CHECK: %[[SUB_FLOAT:[a-zA-Z0-9_]+]] = arith.subf %[[SUB_FLOAT_A]], %[[SUB_FLOAT_B]] : f64
// CHECK: %[[SUB_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_f64 %[[SUB_FLOAT]]
// CHECK: %[[EQ_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[EQ_FLOAT_B:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG1]]
// CHECK: %[[EQ_FLOAT:[a-zA-Z0-9_]+]] = arith.cmpf oeq, %[[EQ_FLOAT_A]], %[[EQ_FLOAT_B]] : f64
// CHECK: %[[EQ_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i1 %[[EQ_FLOAT]]
// CHECK: %[[GE_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[GE_FLOAT_B:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG1]]
// CHECK: %[[GE_FLOAT:[a-zA-Z0-9_]+]] = arith.cmpf oge, %[[GE_FLOAT_A]], %[[GE_FLOAT_B]] : f64
// CHECK: %[[GE_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i1 %[[GE_FLOAT]]
// CHECK: %[[GT_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[GT_FLOAT_B:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG1]]
// CHECK: %[[GT_FLOAT:[a-zA-Z0-9_]+]] = arith.cmpf ogt, %[[GT_FLOAT_A]], %[[GT_FLOAT_B]] : f64
// CHECK: %[[GT_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i1 %[[GT_FLOAT]]
// CHECK: %[[LT_FLOAT_A:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG0]]
// CHECK: %[[LT_FLOAT_B:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[ARG1]]
// CHECK: %[[LT_FLOAT:[a-zA-Z0-9_]+]] = arith.cmpf olt, %[[LT_FLOAT_A]], %[[LT_FLOAT_B]] : f64
// CHECK: %[[LT_FLOAT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i1 %[[LT_FLOAT]]
func.func @float_ops(%arg0: !torch.float, %arg1: !torch.float)
    -> (!torch.float, !torch.float, !torch.float, !torch.float, !torch.float,
        !torch.bool, !torch.bool, !torch.bool, !torch.bool) {
  %add = torch.aten.add.float %arg0, %arg1
      : !torch.float, !torch.float -> !torch.float
  %div = torch.aten.div.float %arg0, %arg1
      : !torch.float, !torch.float -> !torch.float
  %mul = torch.aten.mul.float %arg0, %arg1
      : !torch.float, !torch.float -> !torch.float
  %neg = torch.aten.neg.float %arg0 : !torch.float -> !torch.float
  %sub = torch.aten.sub.float %arg0, %arg1
      : !torch.float, !torch.float -> !torch.float
  %eq = torch.aten.eq.float %arg0, %arg1
      : !torch.float, !torch.float -> !torch.bool
  %ge = torch.aten.ge.float %arg0, %arg1
      : !torch.float, !torch.float -> !torch.bool
  %gt = torch.aten.gt.float %arg0, %arg1
      : !torch.float, !torch.float -> !torch.bool
  %lt = torch.aten.lt.float %arg0, %arg1
      : !torch.float, !torch.float -> !torch.bool
  return %add, %div, %mul, %neg, %sub, %eq, %ge, %gt, %lt
      : !torch.float, !torch.float, !torch.float, !torch.float, !torch.float,
        !torch.bool, !torch.bool, !torch.bool, !torch.bool
}

// CHECK-LABEL: func.func @additional_int_ops(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !torch.int, %[[ARG1:[a-zA-Z0-9_]+]]: !torch.int)
// CHECK: %[[DIV_INT_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[DIV_INT_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[DIV_INT_FLOAT_A:[a-zA-Z0-9_]+]] = arith.sitofp %[[DIV_INT_A]] : i64 to f64
// CHECK: %[[DIV_INT_FLOAT_B:[a-zA-Z0-9_]+]] = arith.sitofp %[[DIV_INT_B]] : i64 to f64
// CHECK: %[[DIV_INT:[a-zA-Z0-9_]+]] = arith.divf %[[DIV_INT_FLOAT_A]], %[[DIV_INT_FLOAT_B]] : f64
// CHECK: %[[DIV_INT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_f64 %[[DIV_INT]]
// CHECK: %[[GE_INT_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[GE_INT_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[GE_INT:[a-zA-Z0-9_]+]] = arith.cmpi sge, %[[GE_INT_A]], %[[GE_INT_B]] : i64
// CHECK: %[[GE_INT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i1 %[[GE_INT]]
// CHECK: %[[GT_INT_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[GT_INT_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[GT_INT:[a-zA-Z0-9_]+]] = arith.cmpi sgt, %[[GT_INT_A]], %[[GT_INT_B]] : i64
// CHECK: %[[GT_INT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i1 %[[GT_INT]]
// CHECK: %[[LT_INT_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[LT_INT_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[LT_INT:[a-zA-Z0-9_]+]] = arith.cmpi slt, %[[LT_INT_A]], %[[LT_INT_B]] : i64
// CHECK: %[[LT_INT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i1 %[[LT_INT]]
// CHECK: %[[NE_INT_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[NE_INT_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[NE_INT:[a-zA-Z0-9_]+]] = arith.cmpi ne, %[[NE_INT_A]], %[[NE_INT_B]] : i64
// CHECK: %[[NE_INT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i1 %[[NE_INT]]
// CHECK: %[[NEG_INT_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[NEG_INT_ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[NEG_INT:[a-zA-Z0-9_]+]] = arith.subi %[[NEG_INT_ZERO]], %[[NEG_INT_A]] : i64
// CHECK: %[[NEG_INT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i64 %[[NEG_INT]]
// CHECK: %[[REM_INT_A:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG0]]
// CHECK: %[[REM_INT_B:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[ARG1]]
// CHECK: %[[REM_QUOTIENT:[a-zA-Z0-9_]+]] = arith.floordivsi %[[REM_INT_A]], %[[REM_INT_B]] : i64
// CHECK: %[[REM_PRODUCT:[a-zA-Z0-9_]+]] = arith.muli %[[REM_QUOTIENT]], %[[REM_INT_B]] : i64
// CHECK: %[[REM_INT:[a-zA-Z0-9_]+]] = arith.subi %[[REM_INT_A]], %[[REM_PRODUCT]] : i64
// CHECK: %[[REM_INT_RESULT:[a-zA-Z0-9_]+]] = torch_c.from_i64 %[[REM_INT]]
func.func @additional_int_ops(%arg0: !torch.int, %arg1: !torch.int)
    -> (!torch.float, !torch.bool, !torch.bool, !torch.bool, !torch.bool,
        !torch.int, !torch.int) {
  %div = torch.aten.div.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.float
  %ge = torch.aten.ge.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.bool
  %gt = torch.aten.gt.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.bool
  %lt = torch.aten.lt.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.bool
  %ne = torch.aten.ne.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.bool
  %neg = torch.aten.neg.int %arg0 : !torch.int -> !torch.int
  %remainder = torch.aten.remainder.int %arg0, %arg1
      : !torch.int, !torch.int -> !torch.int
  return %div, %ge, %gt, %lt, %ne, %neg, %remainder
      : !torch.float, !torch.bool, !torch.bool, !torch.bool, !torch.bool,
        !torch.int, !torch.int
}
