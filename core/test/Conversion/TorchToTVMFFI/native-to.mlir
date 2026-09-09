//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-torch-to-tvm-ffi | FileCheck %s

// CHECK-LABEL: func.func @native_to_tvm_ffi(
// CHECK-SAME: %[[BOOL_ARG:[a-zA-Z0-9_]+]]: i1, %[[INT_ARG:[a-zA-Z0-9_]+]]: i64, %[[FLOAT_ARG:[a-zA-Z0-9_]+]]: f64)
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[BOOL_ARG]] : i1 -> !tvm_ffi.bool
// CHECK: %[[INT:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[INT_ARG]] : i64 -> !tvm_ffi.int
// CHECK: %[[FLOAT:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[FLOAT_ARG]] : f64 -> !tvm_ffi.float
// CHECK: return %[[BOOL]], %[[INT]], %[[FLOAT]] : !tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float
// CHECK-NOT: torch_c.from_i
// CHECK-NOT: torch_c.from_f64
func.func @native_to_tvm_ffi(%arg0: i1, %arg1: i64, %arg2: f64)
    -> (!torch.bool, !torch.int, !torch.float) {
  %bool = torch_c.from_i1 %arg0
  %int = torch_c.from_i64 %arg1
  %float = torch_c.from_f64 %arg2
  return %bool, %int, %float : !torch.bool, !torch.int, !torch.float
}
