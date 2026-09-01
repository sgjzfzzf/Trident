//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

func.func @get_int_to_float(%arg: !torch.int) -> f32 {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!torch.int' to 'f32'}}
  %result = torchext.get %arg : !torch.int -> f32
  return %result : f32
}

// -----

func.func @convert_dtype_to_float(%arg: !torchext.dtype) -> !torch.float {
  // expected-error@+1 {{custom op 'torchext.convert' invalid kind of type specified: expected torch.int, but found '!torch.float'}}
  %result = torchext.convert %arg : !torchext.dtype -> !torch.float
  return %result : !torch.float
}
// -----

func.func @get_requires_bool(%arg: !torch.float) {
  // expected-error@+1 {{'torchext.get' op unsupported get from '!torch.float' to 'i1'}}
  %value = "torchext.get"(%arg) : (!torch.float) -> i1
  return
}
