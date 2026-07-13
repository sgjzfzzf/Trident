//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @test() {
// CHECK-NEXT:    tvm_ffi.return
// CHECK-NEXT:  }
tvm_ffi.func @test() {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: tvm_ffi.func @with_torch_int(
// CHECK-SAME:    %[[ARG:.*]]: !torch.int) -> !torch.int {
// CHECK-NEXT:    tvm_ffi.return %[[ARG]] : !torch.int
// CHECK-NEXT:  }
tvm_ffi.func @with_torch_int(%arg0: !torch.int) -> !torch.int {
  tvm_ffi.return %arg0 : !torch.int
}
