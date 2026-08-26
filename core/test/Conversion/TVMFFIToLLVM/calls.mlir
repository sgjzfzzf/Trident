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
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[NAME:[a-zA-Z0-9_]+]], %[[HANDLE_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[ARG_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][0]
// CHECK: llvm.store %[[ARG]], %[[ARG_SLOT]]
// CHECK: %[[RESULT_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[CALL_ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[CALL_ARG:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[CALL_ARGS]][0]
// CHECK: %[[ARG_COPY:[a-zA-Z0-9_]+]] = llvm.load %[[ARG_SLOT]]
// CHECK: llvm.store %[[ARG_COPY]], %[[CALL_ARG]]
// CHECK: %[[NARGS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[CALL_ARGS]], %[[NARGS]], %[[RESULT_SLOT]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]])
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @global_call(%arg: !tvm_ffi.int) -> !tvm_ffi.int {
  %callee = tvm_ffi.FunctionGetGlobal "test.identity" : !tvm_ffi.function
  %result = tvm_ffi.FunctionCall %callee(%arg)
      : (!tvm_ffi.int) -> !tvm_ffi.int
  return %result : !tvm_ffi.int
}

// -----

// CHECK-LABEL: func.func @local_call(
// CHECK-SAME: %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[NARGS:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[ARG_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][0]
// CHECK: llvm.store %[[ARG]], %[[ARG_SLOT]]
// CHECK: %[[RESULT_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: %[[CONTEXT:[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: call @__tvm_ffi_callee(%[[CONTEXT]], %[[ARGS]], %[[NARGS]], %[[RESULT_SLOT]])
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
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
// CHECK: %[[ARG:[a-zA-Z0-9_]+]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_AUX]][2]
// CHECK: llvm.store %[[ARG]], %[[SOURCE:[a-zA-Z0-9_]+]]
// CHECK: %[[COPY:[a-zA-Z0-9_]+]] = llvm.load %[[SOURCE]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[COPY]], %[[CALL_ARG:[a-zA-Z0-9_]+]]
// CHECK: llvm.call @TVMFFIFunctionCall(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[RESULT_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @exception() -> !tvm_ffi.exception {
  %value = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  return %value : !tvm_ffi.exception
}
