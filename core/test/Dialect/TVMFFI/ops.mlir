//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s --check-prefix=DIALECT

// DIALECT-LABEL: tvm_ffi.func @test() {
// DIALECT-NEXT:    tvm_ffi.return
// DIALECT-NEXT:  }
tvm_ffi.func @test() {
  tvm_ffi.return
}

// -----

// DIALECT-LABEL: func.func @get_bool(
// DIALECT-SAME: [[ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.bool) -> i1 {
// DIALECT: [[VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[ARG]] : !tvm_ffi.bool -> i1
// DIALECT-NEXT: return [[VALUE]] : i1
func.func @get_bool(%arg: !tvm_ffi.bool) -> i1 {
  %value = tvm_ffi.get %arg : !tvm_ffi.bool -> i1
  return %value : i1
}

// DIALECT-LABEL: func.func @get_int(
// DIALECT-SAME: [[INT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> i64 {
// DIALECT-NEXT: [[INT_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[INT_ARG]] : !tvm_ffi.int -> i64
// DIALECT-NEXT: return [[INT_VALUE]] : i64
func.func @get_int(%arg: !tvm_ffi.int) -> i64 {
  %value = tvm_ffi.get %arg : !tvm_ffi.int -> i64
  return %value : i64
}

// DIALECT-LABEL: func.func @get_float(
// DIALECT-SAME: [[FLOAT_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.float) -> f64 {
// DIALECT-NEXT: [[FLOAT_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[FLOAT_ARG]] : !tvm_ffi.float -> f64
// DIALECT-NEXT: return [[FLOAT_VALUE]] : f64
func.func @get_float(%arg: !tvm_ffi.float) -> f64 {
  %value = tvm_ffi.get %arg : !tvm_ffi.float -> f64
  return %value : f64
}

// DIALECT-LABEL: func.func @to_native_scalars(
// DIALECT: [[BOOL:%[a-zA-Z0-9_]+]] = tvm_ffi.to %arg0 : i1 -> !tvm_ffi.bool
// DIALECT: [[INT:%[a-zA-Z0-9_]+]] = tvm_ffi.to %arg1 : i64 -> !tvm_ffi.int
// DIALECT: [[FLOAT:%[a-zA-Z0-9_]+]] = tvm_ffi.to %arg2 : f64 -> !tvm_ffi.float
func.func @to_native_scalars(%bool: i1, %int: i64, %float: f64)
    -> (!tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float) {
  %bool_value = tvm_ffi.to %bool : i1 -> !tvm_ffi.bool
  %int_value = tvm_ffi.to %int : i64 -> !tvm_ffi.int
  %float_value = tvm_ffi.to %float : f64 -> !tvm_ffi.float
  return %bool_value, %int_value, %float_value
      : !tvm_ffi.bool, !tvm_ffi.int, !tvm_ffi.float
}

// DIALECT-LABEL: func.func @get_tensor(
// DIALECT-SAME: [[TENSOR_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.object {
// DIALECT-NEXT: [[TENSOR_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.get [[TENSOR_ARG]] : !tvm_ffi.tensor -> !tvm_ffi.object
// DIALECT-NEXT: return [[TENSOR_VALUE]] : !tvm_ffi.object
func.func @get_tensor(%arg: !tvm_ffi.tensor) -> !tvm_ffi.object {
  %value = tvm_ffi.get %arg : !tvm_ffi.tensor -> !tvm_ffi.object
  return %value : !tvm_ffi.object
}

// DIALECT-LABEL: tvm_ffi.func @with_torch_int(
// DIALECT-SAME:    [[INT_ARG:%[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// DIALECT-NEXT:    tvm_ffi.return [[INT_ARG]] : !torch.int
// DIALECT-NEXT:  }
tvm_ffi.func @with_torch_int(%arg0: !torch.int) -> !torch.int {
  tvm_ffi.return %arg0 : !torch.int
}

// -----

// DIALECT-LABEL: tvm_ffi.func @unordered_union(
// DIALECT-SAME: [[VALUE:%[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.int>) {
// DIALECT-NEXT: tvm_ffi.return
// DIALECT-NEXT: }
tvm_ffi.func @unordered_union(
    %value: !tvm_ffi.union<!tvm_ffi.tensor, !tvm_ffi.int>) {
  tvm_ffi.return
}

// -----

// DIALECT-LABEL: tvm_ffi.func @lifetime_types(
// DIALECT-SAME: [[TENSOR_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor,
// DIALECT-SAME: [[ARRAY_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.array,
// DIALECT-SAME: [[ANY_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.any,
// DIALECT-SAME: [[INDEX_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.tensor {
// DIALECT:      [[ITEM:%[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item [[ARRAY_ARG]][[[INDEX_ARG]]] as !tvm_ffi.tensor : !tvm_ffi.array, !tvm_ffi.int -> !tvm_ffi.tensor
// DIALECT-NEXT: tvm_ffi.ObjectIncRef [[ITEM]] : !tvm_ffi.tensor
// DIALECT-NEXT: tvm_ffi.ObjectDecRef [[ARRAY_ARG]] : !tvm_ffi.array
// DIALECT-NEXT: tvm_ffi.ObjectIncRef [[ANY_ARG]] : !tvm_ffi.any
// DIALECT-NEXT: tvm_ffi.ObjectDecRef [[ANY_ARG]] : !tvm_ffi.any
// DIALECT-NEXT: tvm_ffi.return [[ITEM]] : !tvm_ffi.tensor
// DIALECT-NEXT: }
tvm_ffi.func @lifetime_types(
    %tensor: !tvm_ffi.tensor,
    %array: !tvm_ffi.array,
    %any: !tvm_ffi.any,
    %index: !tvm_ffi.int) -> !tvm_ffi.tensor {
  %item = tvm_ffi.array.get_item %array[%index]
      as !tvm_ffi.tensor
      : !tvm_ffi.array, !tvm_ffi.int
      -> !tvm_ffi.tensor
  tvm_ffi.ObjectIncRef %item : !tvm_ffi.tensor
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.array
  tvm_ffi.ObjectIncRef %any : !tvm_ffi.any
  tvm_ffi.ObjectDecRef %any : !tvm_ffi.any
  tvm_ffi.return %item : !tvm_ffi.tensor
}

// DIALECT-LABEL: tvm_ffi.func @array_unifies_list_and_tuple(
// DIALECT-SAME: [[ARRAY_VALUE:%[a-zA-Z0-9_]+]]: !tvm_ffi.array,
// DIALECT-SAME: [[ARRAY_INDEX:%[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.int {
// DIALECT-NEXT: [[ARRAY_ITEM:%[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item [[ARRAY_VALUE]][[[ARRAY_INDEX]]] as !tvm_ffi.int : !tvm_ffi.array, !tvm_ffi.int -> !tvm_ffi.int
// DIALECT-NEXT: tvm_ffi.return [[ARRAY_ITEM]] : !tvm_ffi.int
// DIALECT-NEXT: }
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

// DIALECT-LABEL: tvm_ffi.func @function_call() -> !tvm_ffi.array {
// DIALECT-NEXT:    [[ARRAY_FUNC:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
// DIALECT-NEXT:    [[ARRAY_RESULT:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall [[ARRAY_FUNC]]() : () -> !tvm_ffi.array
// DIALECT-NEXT:    tvm_ffi.return [[ARRAY_RESULT]] : !tvm_ffi.array
// DIALECT-NEXT:  }
tvm_ffi.func @function_call() -> !tvm_ffi.array {
  %func = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
  %result = tvm_ffi.FunctionCall %func() : () -> !tvm_ffi.array
  tvm_ffi.return %result : !tvm_ffi.array
}

// -----

// A semantic TVM FFI value may be returned through the generic Any ABI slot.
// DIALECT-LABEL: tvm_ffi.func @any_return() -> !tvm_ffi.any {
// DIALECT-NEXT:    [[ANY_VALUE:%[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 1
// DIALECT-NEXT:    tvm_ffi.return [[ANY_VALUE]] : !tvm_ffi.int
// DIALECT-NEXT:  }
tvm_ffi.func @any_return() -> !tvm_ffi.any {
  %value = "tvm_ffi.constant.int"() <{value = 1 : i64}>
      : () -> !tvm_ffi.int
  tvm_ffi.return %value : !tvm_ffi.int
}
