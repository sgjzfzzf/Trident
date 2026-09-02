//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_test.identity_test.identity("test.identity\00")
// CHECK-LABEL: func.func @global_call(
// CHECK-SAME: %[[GLOBAL_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[GLOBAL_NAME:[a-zA-Z0-9_]+]], %[[GLOBAL_HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[GLOBAL_HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[GLOBAL_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[GLOBAL_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[GLOBAL_ARG_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[GLOBAL_ARGS]][0]
// CHECK: llvm.store %[[GLOBAL_ARG]], %[[GLOBAL_ARG_SLOT]]
// CHECK: %[[GLOBAL_RESULT_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[GLOBAL_CALL_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[GLOBAL_CALL_ARG:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[GLOBAL_CALL_ARGS]][0]
// CHECK: %[[GLOBAL_ARG_COPY:[a-zA-Z0-9_]+]] = llvm.load %[[GLOBAL_ARG_SLOT]]
// CHECK: llvm.store %[[GLOBAL_ARG_COPY]], %[[GLOBAL_CALL_ARG]]
// CHECK: %[[GLOBAL_NARGS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[GLOBAL_HANDLE]], %[[GLOBAL_CALL_ARGS]], %[[GLOBAL_NARGS]], %[[GLOBAL_RESULT_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[GLOBAL_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: %[[GLOBAL_RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[GLOBAL_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[GLOBAL_RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @global_call(%arg: !tvm_ffi.int) -> !tvm_ffi.int {
  %callee = tvm_ffi.FunctionGetGlobal "test.identity" : !tvm_ffi.function
  %result = tvm_ffi.FunctionCall %callee(%arg)
      : (!tvm_ffi.int) -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// -----

// CHECK-LABEL: func.func @local_call(
// CHECK-SAME: %[[LOCAL_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[LOCAL_NARGS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[LOCAL_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[LOCAL_ARG_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[LOCAL_ARGS]][0]
// CHECK: llvm.store %[[LOCAL_ARG]], %[[LOCAL_ARG_SLOT]]
// CHECK: %[[LOCAL_RESULT_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[LOCAL_CONTEXT:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: call @__tvm_ffi_callee(%[[LOCAL_CONTEXT]], %[[LOCAL_ARGS]], %[[LOCAL_NARGS]], %[[LOCAL_RESULT_SLOT]])
// CHECK: %[[LOCAL_RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[LOCAL_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[LOCAL_RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func private @__tvm_ffi_callee(
    !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32

func.func @local_call(%arg: !tvm_ffi.int) -> !tvm_ffi.int {
  %result = tvm_ffi.call @callee(%arg) : (!tvm_ffi.int) -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// -----

// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_ExceptionKind_GuardMatch("GuardMatch\00")
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.ffi.Exception_trident.ffi.Exception("trident.ffi.Exception\00")
// CHECK-LABEL: func.func @exception() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[MESSAGE_PTR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_constant_ExceptionKind_GuardMatch
// CHECK: %[[UNDEF:[a-zA-Z0-9_]+]] = llvm.mlir.undef
// CHECK: %[[KIND:[a-zA-Z0-9_]+]] = llvm.mlir.constant(8 : i32) : i32
// CHECK: %[[WITH_KIND:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[KIND]], %[[UNDEF]][0]
// CHECK: %[[AUX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[WITH_AUX:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[AUX]], %[[WITH_KIND]][1]
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.ptrtoint %[[MESSAGE_PTR]] : !llvm.ptr to i64
// CHECK: %[[EXCEPTION_ARG:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: llvm.store %[[EXCEPTION_ARG]], %[[SOURCE:[a-zA-Z0-9_]+]]
// CHECK: %[[COPY:[a-zA-Z0-9_]+]] = llvm.load %[[SOURCE]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY]], %[[CALL_ARG:[a-zA-Z0-9_]+]]
// CHECK: llvm.call @TVMFFIFunctionCall(%[[EXCEPTION_HANDLE:[a-zA-Z0-9_]+]], %[[EXCEPTION_ARGS:[a-zA-Z0-9_]+]], %[[EXCEPTION_NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[EXCEPTION_RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[EXCEPTION_RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @exception() -> !tvm_ffi.exception {
  %value = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  return %value : !tvm_ffi.exception
}

// -----

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
// CHECK: %[[CALL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS]], %[[NARGS]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
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
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[INC_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: %[[DEC_BITS:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[OBJECT]][2]
// CHECK: %[[DEC_HANDLE:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[DEC_BITS]] : i64 to !llvm.ptr
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[DEC_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: return
func.func @object_refs(%object: !tvm_ffi.tensor) {
  tvm_ffi.ObjectIncRef %object : !tvm_ffi.tensor
  tvm_ffi.ObjectDecRef %object : !tvm_ffi.tensor
  return
}

// -----

// CHECK-LABEL: func.func @get_bool(
// CHECK-SAME: %[[BOOL:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> i1 {
// CHECK: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[BOOL]][2]
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = llvm.trunc %[[PAYLOAD]] : i64 to i1
// CHECK-NEXT: return %[[VALUE]] : i1
func.func @get_bool(%value: !tvm_ffi.bool) -> i1 {
  %result = tvm_ffi.get %value : !tvm_ffi.bool -> i1
  return %result : i1
}

// -----

// CHECK-LABEL: func.func @equal(
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[RHS:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> i1 {
// CHECK: %[[LHS_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[RHS_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.store %[[LHS]], %[[LHS_SLOT]]
// CHECK: llvm.store %[[RHS]], %[[RHS_SLOT]]
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE:[a-zA-Z0-9_]+]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RESULT_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[RESULT_SLOT]][0, 2]
// CHECK: %[[RESULT_I64:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_PTR]] : !llvm.ptr -> i64
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.trunc %[[RESULT_I64]] : i64 to i32
// CHECK: %[[ZERO:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[EQUAL:[a-zA-Z0-9_]+]] = arith.cmpi ne, %[[RESULT]], %[[ZERO]] : i32
// CHECK: return %[[EQUAL]] : i1
func.func @equal(%lhs: !tvm_ffi.bool, %rhs: !tvm_ffi.bool) -> i1 {
  %equal = tvm_ffi.eq %lhs, %rhs : !tvm_ffi.bool, !tvm_ffi.bool
  return %equal : i1
}
