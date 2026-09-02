//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @to_native_scalars(
// CHECK-SAME: %[[BOOL_ARG:[a-zA-Z0-9_]+]]: i1, %[[INT_ARG:[a-zA-Z0-9_]+]]: i64, %[[FLOAT_ARG:[a-zA-Z0-9_]+]]: f64)
// CHECK: %[[BOOL_VALUE:[a-zA-Z0-9_]+]] = llvm.zext %[[BOOL_ARG]] : i1 to i64
// CHECK: %[[BOOL_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32)
// CHECK: %[[INT_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32)
// CHECK: %[[FLOAT_VALUE:[a-zA-Z0-9_]+]] = llvm.bitcast %[[FLOAT_ARG]] : f64 to i64
// CHECK: %[[FLOAT_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i32)
func.func @to_native_scalars(%bool: i1, %int: i64, %float: f64)
    -> (!tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float) {
  %bool_value = tvm_ffi.to %bool : i1 -> !tvm_ffi.bool
  %int_value = tvm_ffi.to %int : i64 -> !tvm_ffi.int
  %float_value = tvm_ffi.to %float : f64 -> !tvm_ffi.float
  return %bool_value, %int_value, %float_value
      : !tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float
}
