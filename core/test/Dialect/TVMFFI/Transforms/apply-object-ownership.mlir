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

// Object-valued results in non-object regions still need to be released at
// the region terminator.
// CHECK-LABEL: func.func @object_result_in_guard(
// CHECK:         %[[ITEM:[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item %arg0[%arg1] as !tvm_ffi.tensor
// CHECK:         %[[KEEP:[a-zA-Z0-9_]+]] = arith.constant true
// CHECK:         tvm_ffi.ObjectDecRef %[[ITEM]] : !tvm_ffi.tensor
// CHECK-NEXT:    arithext.and_then.yield %[[KEEP]] : i1
func.func @object_result_in_guard(%arg0: !tvm_ffi.array, %arg1: !tvm_ffi.int) -> i1 {
  %result = "arithext.and_then"() ({
    %item = tvm_ffi.array.get_item %arg0[%arg1]
        as !tvm_ffi.tensor
        : !tvm_ffi.array, !tvm_ffi.int -> !tvm_ffi.tensor
    %keep = arith.constant true
    arithext.and_then.yield %keep : i1
  }) : () -> i1
  return %result : i1
}

// A generic object returned by tvm_ffi.get is a borrowed view and is not
// tracked by this pass.
// CHECK-LABEL: tvm_ffi.func @get_object(
// CHECK-SAME: %arg0: !tvm_ffi.tensor) -> !tvm_ffi.object {
// CHECK-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %arg0 : !tvm_ffi.tensor -> !tvm_ffi.object
// CHECK-NEXT: tvm_ffi.return %[[OBJECT]] : !tvm_ffi.object
tvm_ffi.func @get_object(%arg: !tvm_ffi.tensor) -> !tvm_ffi.object {
  %object = tvm_ffi.get %arg : !tvm_ffi.tensor -> !tvm_ffi.object
  tvm_ffi.return %object : !tvm_ffi.object
}
