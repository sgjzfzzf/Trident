//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @bool_constant() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[VALUE]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @bool_constant() -> !tvm_ffi.bool {
  %value = "tvm_ffi.constant"() <{value = true}> : () -> !tvm_ffi.bool
  return %value : !tvm_ffi.bool
}

// -----

// CHECK-LABEL: func.func @int_constant() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.mlir.constant(-7 : i64) : i64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[VALUE]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @int_constant() -> !tvm_ffi.int {
  %value = "tvm_ffi.constant"() <{value = -7 : i64}> : () -> !tvm_ffi.int
  return %value : !tvm_ffi.int
}

// -----

// CHECK-LABEL: func.func @float_constant() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2.500000e+00 : f64) : f64
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.bitcast %[[VALUE]] : f64 to i64
// CHECK: %[[UNDEF:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[UNDEF]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @float_constant() -> !tvm_ffi.float {
  %value = "tvm_ffi.constant"() <{value = 2.5 : f64}> : () -> !tvm_ffi.float
  return %value : !tvm_ffi.float
}

// -----

// CHECK-LABEL: func.func @dtype_constant() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[PACKED:[a-zA-Z0-9_]+]] = llvm.mlir.constant(73730 : i64) : i64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(5 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[VALUE]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PACKED]], %[[WITH_AUX]][2]
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @dtype_constant() -> !tvm_ffi.dtype {
  %value = "tvm_ffi.constant"() <{value = [2, 32, 1]}>
      : () -> !tvm_ffi.dtype
  return %value : !tvm_ffi.dtype
}

// -----

// CHECK-LABEL: func.func @none_constant() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[VALUE]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @none_constant() -> !tvm_ffi.none {
  %value = "tvm_ffi.constant"() <{value}> : () -> !tvm_ffi.none
  return %value : !tvm_ffi.none
}

// -----

// CHECK-LABEL: func.func @cast(
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>)
// CHECK-NOT: tvm_ffi.cast
// CHECK: return %[[VALUE]] : !llvm.struct<(i32, i32, i64)>
func.func @cast(%value: !tvm_ffi.int) -> !tvm_ffi.any {
  %result = tvm_ffi.cast %value : !tvm_ffi.int -> !tvm_ffi.any
  return %result : !tvm_ffi.any
}
