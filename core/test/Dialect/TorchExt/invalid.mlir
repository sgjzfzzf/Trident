//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

module {
  func.func @specializations_size(%tensor: !torch.vtensor<[4],f32>, %value: !torch.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'torchext.trident_kernel_launch' op specializations and kernel operands must have the same size}}
    "torchext.trident_kernel_launch"(%one, %one, %one, %one, %one, %one, %tensor, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2>, specializations = [#torchext.variable_specialization<kind = !llvm.ptr, divisibility = 16>]}> : (i64, i64, i64, i64, i64, i64, !torch.vtensor<[4],f32>, !torch.int) -> ()
    func.return
  }
}

// -----

module {
  func.func @missing_specialization(%value: !torch.int) {
    %one = arith.constant 1 : i64
    // expected-error@+4 {{expected attribute value}}
    torchext.trident_kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%value : !torch.int)
    func.return
  }
}

// -----

module {
  func.func @invalid_constant_specialization_type(%value: !torch.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{constant specialization requires an i1, i64, f64, string, or tuple value}}
    "torchext.trident_kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>, specializations = [#torchext.constant_specialization<value = 1 : i32>]}> : (i64, i64, i64, i64, i64, i64, !torch.int) -> ()
    func.return
  }
}

// -----

module {
  func.func @invalid_specialization_element(%value: !torch.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'torchext.trident_kernel_launch' op attribute 'specializations' failed to satisfy constraint: array of Triton argument specializations}}
    "torchext.trident_kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>, specializations = [42 : i64]}> : (i64, i64, i64, i64, i64, i64, !torch.int) -> ()
    func.return
  }
}

// -----

module {
  func.func @eq_operand_type_mismatch(%lhs: !torch.int, %rhs: !torch.float) {
    // expected-error@+1 {{'torchext.eq' op requires all operands to have the same type}}
    %equal = "torchext.eq"(%lhs, %rhs) : (!torch.int, !torch.float) -> i1
    func.return
  }
}
