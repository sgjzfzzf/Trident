//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that torch.aten.t (transpose view) is lowered through the
// AtenGen FFI dispatch path and exposed through the generated TVM FFI wrapper.
//
// NOTE: aten.t is a view op — its result aliases the operand's storage. This
// test only checks that the FFI dispatch path is generated correctly; the
// runtime semantics of the transposed view (strides preserved across the
// DLPack boundary) are exercised end-to-end by test/test_t.py.

// CHECK-LABEL: llvm.func @torch.aten.t(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[GETGLOBAL:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[FUNCTION_NAME:[a-zA-Z0-9_]+]], %[[HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[ARGS_COPY:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RET_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[RET:[a-zA-Z0-9_]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_t(
// CHECK-SAME: %[[WRAP_CONTEXT:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAP_ARGS:[a-zA-Z0-9_]+]]: !llvm.ptr, %[[WRAP_NARGS:[a-zA-Z0-9_]+]]: i32, %[[WRAP_RESULT:[a-zA-Z0-9_]+]]: !llvm.ptr) -> i32 {
// CHECK: %[[WRAP_ARG:[a-zA-Z0-9_]+]] = llvm.load %[[WRAP_ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAP_RET:[a-zA-Z0-9_]+]] = llvm.call @t(%[[WRAP_ARG]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[WRAP_RET]], %[[WRAP_RESULT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
func.func @torch.aten.t(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

tvm_ffi.func @t(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  tvm_ffi.return %0 : !torch.vtensor<[3,2],f32>
}
