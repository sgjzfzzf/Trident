//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @cast_rejects_incompatible_union(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{'tvm_ffi.cast' op operand type '!tvm_ffi.int' and result type '!tvm_ffi.union<!tvm_ffi.float, !tvm_ffi.exception>' are cast incompatible}}
  %result = tvm_ffi.cast %arg
      : !tvm_ffi.int
      -> !tvm_ffi.union<!tvm_ffi.float, !tvm_ffi.exception>
  return
}

// -----

func.func @cast_rejects_torch_operand(%arg: !torch.int) {
  // expected-error@+1 {{'tvm_ffi.cast' op operand #0 must be TVM FFI value with a TVMFFIAny ABI representation, but got '!torch.int'}}
  %result = "tvm_ffi.cast"(%arg)
      : (!torch.int) -> !tvm_ffi.any
  return
}

// -----

func.func @cast_rejects_concrete_result(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{'tvm_ffi.cast' op result #0 must be widened TVM FFI any or union value, but got '!tvm_ffi.int'}}
  %result = "tvm_ffi.cast"(%arg)
      : (!tvm_ffi.int) -> !tvm_ffi.int
  return
}

// -----

// expected-error@+1 {{union requires at least two member types}}
func.func @single_member_union(%arg: !tvm_ffi.union<!tvm_ffi.int>)

// -----

// expected-error@+2 {{union member types must be unique}}
func.func @duplicate_union_member(
    %arg: !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.int>)

// -----

func.func @invalid_dtype_constant() {
  // expected-error@+1 {{dtype result requires [code, bits, lanes]}}
  %value = "tvm_ffi.constant.dtype"() <{value = [1, 2]}> : () -> !tvm_ffi.dtype
  return
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
  %value = "tvm_ffi.constant.float"() <{value = 1.0 : f64}>
      : () -> !tvm_ffi.float
  // expected-error@+1 {{operand type is not a member of the result type}}
  tvm_ffi.return %value : !tvm_ffi.float
}
// -----

func.func @get_int_width(%arg: !tvm_ffi.int) {
  // expected-error@+1 {{'tvm_ffi.get' op unsupported get from '!tvm_ffi.int' to 'i128'}}
  %value = "tvm_ffi.get"(%arg) : (!tvm_ffi.int) -> i128
  return
}

// -----

func.func @get_unsupported_array(%arg: !tvm_ffi.array) {
  // expected-error@+1 {{'tvm_ffi.get' op unsupported get from '!tvm_ffi.array' to 'i64'}}
  %value = "tvm_ffi.get"(%arg) : (!tvm_ffi.array) -> i64
  return
}

// -----

func.func @to_mismatched_native_type(%value: i64) {
  // expected-error@+1 {{'tvm_ffi.to' op unsupported native-to-TVM FFI conversion from 'i64' to '!tvm_ffi.bool'}}
  %result = tvm_ffi.to %value : i64 -> !tvm_ffi.bool
  return
}
