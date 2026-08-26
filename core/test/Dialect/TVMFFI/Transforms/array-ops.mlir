//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -decompose-tvm-ffi | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @create(
// CHECK:         %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
// CHECK-NEXT:    %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %arg1) : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.array
// CHECK-NEXT:    tvm_ffi.ObjectDecRef %arg0 : !tvm_ffi.tensor
// CHECK-NEXT:    tvm_ffi.return %[[RESULT]] : !tvm_ffi.array
tvm_ffi.func @create(%arg0: !tvm_ffi.tensor, %arg1: !tvm_ffi.int) -> !tvm_ffi.array {
  %array = "tvm_ffi.array.create"(%arg0, %arg1)
      : (!tvm_ffi.tensor, !tvm_ffi.int) -> !tvm_ffi.array
  tvm_ffi.return %array : !tvm_ffi.array
}

// -----

// CHECK-LABEL: tvm_ffi.func @get_item(
// CHECK:         %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.ArrayGetItem" : !tvm_ffi.function
// CHECK-NEXT:    %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %arg1) : (!tvm_ffi.array, !tvm_ffi.int) -> !tvm_ffi.int
// CHECK-NEXT:    tvm_ffi.return %[[RESULT]] : !tvm_ffi.int
tvm_ffi.func @get_item(%arg0: !tvm_ffi.array, %arg1: !tvm_ffi.int) -> !tvm_ffi.int {
  %item = tvm_ffi.array.get_item %arg0[%arg1]
      as !tvm_ffi.int
      : !tvm_ffi.array, !tvm_ffi.int -> !tvm_ffi.int
  tvm_ffi.return %item : !tvm_ffi.int
}
