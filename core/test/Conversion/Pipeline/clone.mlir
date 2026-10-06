//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// An explicit clone call is preserved through canonicalization and
// reaches the name-based TVM FFI dispatch path with its memory format operand.
// In particular, the contiguous-memory-format clone must not fold to its input:
// the Python regression test passes a transposed tensor here and checks that a
// distinct, contiguous allocation is returned.

// CHECK-LABEL: llvm.func @aten.clone(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// The callee is resolved once at load time, so the dispatch reads the cached
// handle rather than calling TVMFFIFunctionGetGlobal per invocation. The
// module owns the cached reference, so each call retains it and releases it
// again below; the two must stay paired or the handle is over-released.
// CHECK-NOT: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: %[[HANDLE_ADDR:[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_trident.aten.clone : !llvm.ptr
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIObjectIncRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[ARGS_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ARG0]], %[[ARGS]]
// CHECK: %[[FORMAT_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][1]
// CHECK: llvm.store %[[FORMAT:[a-zA-Z0-9_]+]], %[[FORMAT_SLOT]]
// CHECK: %[[RET_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[CALL_ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// The call site retains the cached handle, so it also releases it.
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]]) : (!llvm.ptr) -> i32
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_clone(
// CHECK-SAME: %[[WRAP_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAP_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAP_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAP_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr)
// CHECK: %[[WRAP_ARG:[a-zA-Z0-9_]+]] = llvm.load %[[WRAP_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAP_RET:[a-zA-Z0-9_]+]] = llvm.call @clone(%[[WRAP_ARG]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[WRAP_RET]], %[[WRAP_RESULT]]
// CHECK-LABEL: llvm.func @__tvm_ffi_clone_preserve(
// CHECK-SAME: %[[PRESERVE_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[PRESERVE_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[PRESERVE_NARGS:[a-zA-Z0-9_]+]]: i32, %[[PRESERVE_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr)
// CHECK: %[[PRESERVE_ARG:[a-zA-Z0-9_]+]] = llvm.load %[[PRESERVE_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[PRESERVE_RET:[a-zA-Z0-9_]+]] = llvm.call @clone_preserve(%[[PRESERVE_ARG]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[PRESERVE_RET]], %[[PRESERVE_RESULT]]

func.func @aten.clone(%arg0: !tvm_ffi.tensor)
    -> !tvm_ffi.tensor {
  %memory_format = tvm_ffi.constant.int 0
  %clone_handle, %clone_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.clone" : !tvm_ffi.function, i1
  cf.assert %clone_lookup, "lookup failed"
  %clone, %clone_status = tvm_ffi.FunctionCall %clone_handle(%arg0, %memory_format) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tensor, i1
  cf.assert %clone_status, "call failed"
  return %clone : !tvm_ffi.tensor
}

tvm_ffi.func @clone(%arg0: !tvm_ffi.tensor)
    -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %memory_format = tvm_ffi.constant.int 0
  %clone_handle, %clone_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.clone" : !tvm_ffi.function, i1
  cf.assert %clone_lookup, "lookup failed"
  %clone, %clone_status = tvm_ffi.FunctionCall %clone_handle(%arg0, %memory_format) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tensor, i1
  cf.assert %clone_status, "call failed"
  tvm_ffi.return %clone : !tvm_ffi.tensor
}

// Preserve-format clone is covered separately so the runtime test can verify
// that the memory-format operand is not merely present but also honored.
tvm_ffi.func @clone_preserve(%arg0: !tvm_ffi.tensor)
    -> !tvm_ffi.tensor attributes {emit_tvm_ffi_abi} {
  %memory_format = tvm_ffi.constant.int 1
  %clone_handle, %clone_lookup = tvm_ffi.FunctionGetGlobal "trident.aten.clone" : !tvm_ffi.function, i1
  cf.assert %clone_lookup, "lookup failed"
  %clone, %clone_status = tvm_ffi.FunctionCall %clone_handle(%arg0, %memory_format) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.tensor, i1
  cf.assert %clone_status, "call failed"
  tvm_ffi.return %clone : !tvm_ffi.tensor
}
