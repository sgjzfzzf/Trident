//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics



module {
  func.func @specializations_size(%tensor: !tvm_ffi.tensor, %value: !tvm_ffi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'tvm_ffi.kernel_launch' op specializations and kernel operands must have the same size}}
    "tvm_ffi.kernel_launch"(%one, %one, %one, %one, %one, %one, %tensor, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2>, specializations = [#tvm_ffi.variable_specialization<kind = !llvm.ptr, divisibility = 16>]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi.tensor, !tvm_ffi.int) -> ()
    func.return
  }
}

// -----

module {
  func.func @missing_specialization(%value: !tvm_ffi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+4 {{expected attribute value}}
    tvm_ffi.kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%value : !tvm_ffi.int)
    func.return
  }
}

// -----

module {
  func.func @invalid_constant_specialization_type(%value: !tvm_ffi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{constant specialization requires an i1, i64, f64, string, or tuple value}}
    "tvm_ffi.kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>, specializations = [#tvm_ffi.constant_specialization<value = 1 : i32>]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi.int) -> ()
    func.return
  }
}

// -----

module {
  func.func @invalid_specialization_element(%value: !tvm_ffi.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'tvm_ffi.kernel_launch' op attribute 'specializations' failed to satisfy constraint: array of Triton argument specializations}}
    "tvm_ffi.kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>, specializations = [42 : i64]}> : (i64, i64, i64, i64, i64, i64, !tvm_ffi.int) -> ()
    func.return
  }
}
