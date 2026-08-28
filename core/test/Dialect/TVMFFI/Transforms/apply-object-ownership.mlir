//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -apply-object-ownership | FileCheck %s

// The ownership pass retains object-valued return operands for the caller.
// CHECK-LABEL: tvm_ffi.func @return_tensor(
// CHECK-SAME: [[ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK-NEXT: tvm_ffi.ObjectIncRef [[ARG]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.return [[ARG]] : !tvm_ffi.tensor
tvm_ffi.func @return_tensor(%arg: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  tvm_ffi.return %arg : !tvm_ffi.tensor
}
