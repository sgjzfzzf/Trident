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

// -----

// CHECK-LABEL: tvm_ffi.func @lifetime_types(
// CHECK-SAME: !tvm_ffi.tensor
// CHECK-SAME: !tvm_ffi.array
tvm_ffi.func @lifetime_types(
    %tensor: !tvm_ffi.tensor,
    %array: !tvm_ffi.array,
    %index: !tvm_ffi.int) -> !tvm_ffi.tensor {
  // CHECK: %[[ITEM:.*]] = tvm_ffi.array.get_item %arg1[%arg2] as !tvm_ffi.tensor
  %item = tvm_ffi.array.get_item %array[%index]
      as !tvm_ffi.tensor
      : !tvm_ffi.array, !tvm_ffi.int
      -> !tvm_ffi.tensor
  // CHECK: tvm_ffi.ObjectIncRef %[[ITEM]] : !tvm_ffi.tensor
  tvm_ffi.ObjectIncRef %item : !tvm_ffi.tensor
  // CHECK: tvm_ffi.ObjectDecRef %arg1 : !tvm_ffi.array
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.array
  tvm_ffi.return %item : !tvm_ffi.tensor
}

// CHECK-LABEL: tvm_ffi.func @array_unifies_list_and_tuple
tvm_ffi.func @array_unifies_list_and_tuple(
    %list_or_tuple: !tvm_ffi.array,
    %index: !tvm_ffi.int) -> !tvm_ffi.int {
  %item = tvm_ffi.array.get_item %list_or_tuple[%index]
      as !tvm_ffi.int
      : !tvm_ffi.array, !tvm_ffi.int
      -> !tvm_ffi.int
  tvm_ffi.return %item : !tvm_ffi.int
}

// -----

// CHECK-LABEL: tvm_ffi.func @function_call() -> !tvm_ffi.array {
// CHECK-NEXT:    %[[FUNC:.*]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
// CHECK-NEXT:    %[[RESULT:.*]] = tvm_ffi.FunctionCall %[[FUNC]]() : () -> !tvm_ffi.array
// CHECK-NEXT:    tvm_ffi.return %[[RESULT]] : !tvm_ffi.array
// CHECK-NEXT:  }
tvm_ffi.func @function_call() -> !tvm_ffi.array {
  %func = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
  %result = tvm_ffi.FunctionCall %func() : () -> !tvm_ffi.array
  tvm_ffi.return %result : !tvm_ffi.array
}
