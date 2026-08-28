//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{union requires at least two member types}}
func.func @single_member_union(%arg: !tvm_ffi.union<!tvm_ffi.int>)

// -----

// expected-error@+2 {{union member types must be unique}}
func.func @duplicate_union_member(
    %arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.int>)

// -----

func.func @invalid_bool_constant() {
  // expected-error@+1 {{'tvm_ffi.constant' op bool result requires a BoolAttr}}
  %value = "tvm_ffi.constant"() <{value = 1 : i64}> : () -> !tvm_ffi.bool
  return
}

// -----

func.func @array_get_item_missing_element_type(
    %array: !tvm_ffi.array, %index: !tvm_ffi.int) -> !tvm_ffi.int {
  // expected-error@+1 {{'tvm_ffi.array.get_item' op array element type must be specified}}
  %item = tvm_ffi.array.get_item %array[%index]
      : !tvm_ffi.array, !tvm_ffi.int -> !tvm_ffi.int
  return %item : !tvm_ffi.int
}

// -----

// expected-error@+1 {{'tvm_ffi.func' op must have public visibility}}
tvm_ffi.func private @private_function() {
  tvm_ffi.return
}

// -----

tvm_ffi.func @empty_any_return() -> !tvm_ffi.any {
  // expected-error@+1 {{'tvm_ffi.return' op an any or union function must return at least one value}}
  tvm_ffi.return
}

// -----

tvm_ffi.func @invalid_union_return()
    -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.bool> {
  %value = "tvm_ffi.constant"() <{value = 1.0 : f64}>
      : () -> !tvm_ffi.float
  // expected-error@+1 {{operand type is not a member of the result type}}
  tvm_ffi.return %value : !tvm_ffi.float
}
// -----

func.func @get_requires_bool(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{'tvm_ffi.get' op operand #0 must be , but got '!tvm_ffi.int'}}
  %value = "tvm_ffi.get"(%arg) : (!tvm_ffi.int) -> i1
  return
}
