//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @get_int_to_i32(%arg: !torch.int) -> i32 {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!torch.int' to 'i32'}}
  %result = torchext.get %arg : !torch.int -> i32
  return %result : i32
}

// -----

func.func @convert_dtype_to_float(%arg: !torchext.dtype) -> !torch.float {
  // expected-error@+1 {{custom op 'torchext.convert' invalid kind of type specified: expected torch.int, but found '!torch.float'}}
  %result = torchext.convert %arg : !torchext.dtype -> !torch.float
  return %result : !torch.float
}
// -----

func.func @get_float_to_f32(%arg: !torch.float) -> f32 {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!torch.float' to 'f32'}}
  %result = torchext.get %arg : !torch.float -> f32
  return %result : f32
}

// -----

func.func @get_requires_bool(%arg: !torch.float) {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!torch.float' to 'i1'}}
  %value = "torchext.get"(%arg) : (!torch.float) -> i1
  return
}

// -----

func.func @get_bool_to_i8(%arg: !torch.bool) -> i8 {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!torch.bool' to 'i8'}}
  %value = "torchext.get"(%arg) : (!torch.bool) -> i8
  return %value : i8
}

// -----

func.func @get_tvm_int_to_float(%arg: !tvm_ffi.int) -> f32 {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!tvm_ffi.int' to 'f32'}}
  %value = "torchext.get"(%arg) : (!tvm_ffi.int) -> f32
  return %value : f32
}

// -----

func.func @get_tvm_tensor_to_float(%arg: !tvm_ffi.tensor) -> f64 {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!tvm_ffi.tensor' to 'f64'}}
  %value = "torchext.get"(%arg) : (!tvm_ffi.tensor) -> f64
  return %value : f64
}

// -----

func.func @get_tvm_array_to_float(%arg: !tvm_ffi.array) -> f64 {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!tvm_ffi.array' to 'f64'}}
  %value = "torchext.get"(%arg) : (!tvm_ffi.array) -> f64
  return %value : f64
}
