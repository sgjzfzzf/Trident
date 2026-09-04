//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -ownership-deallocation -verify-diagnostics

module {
  func.func @structured_control_flow(%cond: i1) -> i32 {
    // expected-error @+1 {{nested regions must be lowered to CFG before ownership deallocation}}
    %result = scf.if %cond -> i32 {
      %value = arith.constant 0 : i32
      scf.yield %value : i32
    } else {
      %value = arith.constant 1 : i32
      scf.yield %value : i32
    }
    return %result : i32
  }
}

// -----

module {
  func.func @consume_borrowed_handle(%function: !tvm_ffi.function) {
    // expected-error @+1 {{object reference count would become negative}}
    tvm_ffi.FunctionCall %function() : () -> ()
    return
  }
}
