//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @tensor_literal() -> !llvm.struct<(i32, i32, i64)> {
// CHECK: llvm.call @aoti_torch_create_tensor_from_blob
// CHECK: llvm.call @aoti_torch_empty_strided
// CHECK: llvm.call @aoti_torch_copy_
// CHECK: llvm.call @aoti_torch_delete_tensor_object
// CHECK: llvm.call @TVMFFIFunctionCall(%[[HANDLE:[a-zA-Z0-9_]+]], %[[ARGS:[a-zA-Z0-9_]+]], %[[NARGS:[a-zA-Z0-9_]+]], %[[RESULT_SLOT:[a-zA-Z0-9_]+]])
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = llvm.load %[[RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: return %[[RESULT]] : !llvm.struct<(i32, i32, i64)>
func.func @tensor_literal() -> !tvm_ffi.tensor {
  %literal = tvm_ffi.tensor.literal dense<[1.0, 2.0]> : tensor<2xf32>
      : !tvm_ffi.tensor
  return %literal : !tvm_ffi.tensor
}
