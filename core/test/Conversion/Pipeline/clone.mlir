//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// A concrete torch.aten.clone is generalized before any folding pass and
// reaches the name-based TVM FFI dispatch path with its memory format operand.
// In particular, the contiguous-memory-format clone must not fold to its input:
// the Python regression test passes a transposed tensor here and checks that a
// distinct, contiguous allocation is returned.

// CHECK-LABEL: llvm.func @torch.aten.clone(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[GETGLOBAL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[FUNCTION_NAME:[a-zA-Z0-9_]+]], %[[HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[ARGS:[a-zA-Z0-9_]+]] = llvm.alloca %[[ARGS_COUNT:[a-zA-Z0-9_]+]] x !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ARG0]], %[[ARGS]]
// CHECK: %[[FORMAT_SLOT:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[ARGS]][1]
// CHECK: llvm.store %[[FORMAT:[a-zA-Z0-9_]+]], %[[FORMAT_SLOT]]
// CHECK: %[[RET_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[CALL_ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
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

func.func @torch.aten.clone(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> {
  %memory_format = torch.constant.int 0
  %clone = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  return %clone : !torch.vtensor<[32,2],f32>
}

tvm_ffi.func @clone(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> attributes {emit_tvm_ffi_abi} {
  %memory_format = torch.constant.int 0
  %clone = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  tvm_ffi.return %clone : !torch.vtensor<[32,2],f32>
}

// Preserve-format clone is covered separately so the runtime test can verify
// that the memory-format operand is not merely present but also honored.
tvm_ffi.func @clone_preserve(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> attributes {emit_tvm_ffi_abi} {
  %memory_format = torch.constant.int 1
  %clone = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  tvm_ffi.return %clone : !torch.vtensor<[32,2],f32>
}
