//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -ownership-deallocation | FileCheck %s --check-prefix=OWNERSHIP

// Object-valued return operands are retained for the caller before local
// ownership credits are released.
// OWNERSHIP-LABEL: tvm_ffi.func @return_tensor(
// OWNERSHIP-SAME: [[RETURN_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// OWNERSHIP-NEXT: tvm_ffi.ObjectIncRef [[RETURN_ARG]] : !tvm_ffi.tensor
// OWNERSHIP-NEXT: tvm_ffi.return [[RETURN_ARG]] : !tvm_ffi.tensor
tvm_ffi.func @return_tensor(%arg: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  tvm_ffi.return %arg : !tvm_ffi.tensor
}

// A generic object returned by tvm_ffi.get is a borrowed view and is not
// tracked by this pass.
// OWNERSHIP-LABEL: tvm_ffi.func @get_object(
// OWNERSHIP-SAME: %[[OBJECT_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.object {
// OWNERSHIP-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[OBJECT_ARG]] : !tvm_ffi.tensor -> !tvm_ffi.object
// OWNERSHIP-NEXT: tvm_ffi.return %[[OBJECT]] : !tvm_ffi.object
tvm_ffi.func @get_object(%arg: !tvm_ffi.tensor) -> !tvm_ffi.object {
  %object = tvm_ffi.get %arg : !tvm_ffi.tensor -> !tvm_ffi.object
  tvm_ffi.return %object : !tvm_ffi.object
}
