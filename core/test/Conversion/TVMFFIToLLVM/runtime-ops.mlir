//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIObjectIncRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_ffi.ArraySize_ffi.ArraySize("ffi.ArraySize\00")
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_ffi.StructuralEqual_ffi.StructuralEqual("ffi.StructuralEqual\00")
// CHECK-LABEL: func.func @array_length(
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> i64 {
// CHECK: %[[SOURCE:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.store %[[ARRAY]], %[[SOURCE]]
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[ARG_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][0]
// CHECK: %[[ARG:[a-zA-Z0-9_]+]] = llvm.load %[[SOURCE]]
// CHECK: llvm.store %[[ARG]], %[[ARG_SLOT]]
// CHECK: %[[NARGS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_SLOT:[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[CALL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS]], %[[NARGS]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]])
// CHECK: %[[PAYLOAD_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[RESULT_SLOT]][0, 2]
// CHECK: %[[LENGTH:[a-zA-Z0-9_]+]] = llvm.load %[[PAYLOAD_PTR]] : !llvm.ptr -> i64
// CHECK: return %[[LENGTH]] : i64
func.func @array_length(%array: !tvm_ffi.array) -> i64 {
  %length = tvm_ffi.array.length %array : !tvm_ffi.array
  return %length : i64
}

// -----

// CHECK-LABEL: func.func @object_refs(
// CHECK-SAME: %[[OBJECT:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK: %[[INC_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OBJECT]][2]
// CHECK: %[[INC_HANDLE:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[INC_BITS]] : i64 to !llvm.ptr
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[INC_HANDLE]])
// CHECK: %[[DEC_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OBJECT]][2]
// CHECK: %[[DEC_HANDLE:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[DEC_BITS]] : i64 to !llvm.ptr
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[DEC_HANDLE]])
// CHECK: return
func.func @object_refs(%object: !tvm_ffi.tensor) {
  tvm_ffi.ObjectIncRef %object : !tvm_ffi.tensor
  tvm_ffi.ObjectDecRef %object : !tvm_ffi.tensor
  return
}

// -----

// CHECK-LABEL: func.func @equal(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[RHS:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> i1 {
// CHECK: %[[LHS_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[RHS_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.store %[[LHS]], %[[LHS_SLOT]]
// CHECK: llvm.store %[[RHS]], %[[RHS_SLOT]]
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: %[[RESULT_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %{{[a-zA-Z0-9_]+}}[0, 2]
// CHECK: %[[RESULT_I64:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_PTR]] : !llvm.ptr -> i64
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.trunc %[[RESULT_I64]] : i64 to i32
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[EQUAL:[a-zA-Z0-9_]+]] = arith.cmpi ne, %[[RESULT]], %[[ZERO]] : i32
// CHECK: return %[[EQUAL]] : i1
func.func @equal(%lhs: !tvm_ffi.bool, %rhs: !tvm_ffi.bool) -> i1 {
  %equal = tvm_ffi.eq %lhs, %rhs : !tvm_ffi.bool, !tvm_ffi.bool
  return %equal : i1
}
