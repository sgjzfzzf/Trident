//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -decompose-tvm-ffi | FileCheck %s --check-prefix=DECOMPOSE
// RUN: trident-core-opt %s -split-input-file -convert-arith-ext-to-scf -convert-scf-to-cf -ownership-deallocation | FileCheck %s --check-prefix=OWNERSHIP

// Array construction and indexed access are decomposed into runtime calls.
// DECOMPOSE-LABEL: tvm_ffi.func @create(
// DECOMPOSE: %[[ARRAY_FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
// DECOMPOSE-NEXT: %[[ARRAY_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[ARRAY_FUNC]](%arg0, %arg1) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.array
// DECOMPOSE-NEXT: tvm_ffi.return %[[ARRAY_RESULT]] : !tvm_ffi.array
tvm_ffi.func @create(%arg0: !tvm_ffi.tensor, %arg1: !tvm_ffi.int) -> !tvm_ffi.array {
  %array = "tvm_ffi.array.create"(%arg0, %arg1)
      : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.array
  tvm_ffi.return %array : !tvm_ffi.array
}

// DECOMPOSE-LABEL: tvm_ffi.func @get_item(
// DECOMPOSE: %[[ITEM_FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.function
// DECOMPOSE-NEXT: %[[ITEM_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[ITEM_FUNC]](%arg0, %arg1) : (!tvm_ffi.array, !tvm_ffi.int) -> !tvm_ffi.int
// DECOMPOSE-NEXT: tvm_ffi.return %[[ITEM_RESULT]] : !tvm_ffi.int
tvm_ffi.func @get_item(%arg0: !tvm_ffi.array, %arg1: !tvm_ffi.int) -> !tvm_ffi.int {
  %item = tvm_ffi.array.get_item %arg0[%arg1]
      as !tvm_ffi.int
      : !tvm_ffi.array, !tvm_ffi.int -> !tvm_ffi.int
  tvm_ffi.return %item : !tvm_ffi.int
}

// The ownership pass retains object-valued return operands for the caller.
// OWNERSHIP-LABEL: tvm_ffi.func @return_tensor(
// OWNERSHIP-SAME: [[RETURN_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// OWNERSHIP-NEXT: tvm_ffi.ObjectIncRef [[RETURN_ARG]] : !tvm_ffi.tensor
// OWNERSHIP-NEXT: tvm_ffi.return [[RETURN_ARG]] : !tvm_ffi.tensor
tvm_ffi.func @return_tensor(%arg: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  tvm_ffi.return %arg : !tvm_ffi.tensor
}

// Object-valued results in control flow are released on the outgoing CFG edge.
// OWNERSHIP-LABEL: func.func @object_result_in_guard(
// OWNERSHIP: %[[ITEM:[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item %arg0[%arg1] as !tvm_ffi.tensor
// OWNERSHIP: tvm_ffi.ObjectDecRef %[[ITEM]] : !tvm_ffi.tensor
// OWNERSHIP: return
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
// OWNERSHIP-LABEL: tvm_ffi.func @get_object(
// OWNERSHIP-SAME: %[[OBJECT_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.object {
// OWNERSHIP-NEXT: %[[OBJECT:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[OBJECT_ARG]] : !tvm_ffi.tensor -> !tvm_ffi.object
// OWNERSHIP-NEXT: tvm_ffi.return %[[OBJECT]] : !tvm_ffi.object
tvm_ffi.func @get_object(%arg: !tvm_ffi.tensor) -> !tvm_ffi.object {
  %object = tvm_ffi.get %arg : !tvm_ffi.tensor -> !tvm_ffi.object
  tvm_ffi.return %object : !tvm_ffi.object
}
