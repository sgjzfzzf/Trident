//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file --convert-tvm-ffi-to-gpu -verify-diagnostics

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  tvm_ffi.func @invalid_result(%value: !tvm_ffi.int) -> !tvm_ffi.int {
    %one = arith.constant 1 : i64
    // expected-error@+2 {{'tvm_ffi.kernel_launch' op enclosing function result must be !tvm_ffi.any or contain !tvm_ffi.exception}}
    // expected-error@+1 {{failed to legalize operation 'tvm_ffi.kernel_launch' that was explicitly marked illegal}}
    tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%value : !tvm_ffi.int #tvm_ffi.variable_specialization<kind = i32>)
    %result = tvm_ffi.constant.int 0
    tvm_ffi.return %result : !tvm_ffi.int
  }
}
