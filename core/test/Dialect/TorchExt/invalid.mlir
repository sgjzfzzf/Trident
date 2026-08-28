//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @cast_int_to_float(%arg: !torch.int) -> f32 {
  // expected-error@+1 {{'torchext.cast' op operand type '!torch.int' and result type 'f32' are cast incompatible}}
  %result = torchext.cast %arg : !torch.int -> f32
  return %result : f32
}

// -----

func.func @convert_dtype_to_float(%arg: !torchext.dtype) -> !torch.float {
  // expected-error@+1 {{custom op 'torchext.convert' invalid kind of type specified: expected torch.int, but found '!torch.float'}}
  %result = torchext.convert %arg : !torchext.dtype -> !torch.float
  return %result : !torch.float
}
// -----

func.func @get_requires_bool(%arg: !torch.int) {
  // expected-error@+1 {{'torchext.get' op operand #0 must be Torch BoolType}}
  %value = "torchext.get"(%arg) : (!torch.int) -> i1
  return
}
