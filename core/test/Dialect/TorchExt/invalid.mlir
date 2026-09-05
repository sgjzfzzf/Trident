//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -verify-diagnostics

module {
  func.func @arg_attrs_size(%tensor: !torch.vtensor<[4],f32>, %value: !torch.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'torchext.trident_kernel_launch' op arg_attrs and kernel operands must have the same size}}
    "torchext.trident_kernel_launch"(%one, %one, %one, %one, %one, %one, %tensor, %value) <{arg_attrs = [{triton.specialization = #torchext.specialization<kind = !llvm.ptr, divisibility = 16>}], kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2>}> : (i64, i64, i64, i64, i64, i64, !torch.vtensor<[4],f32>, !torch.int) -> ()
    func.return
  }
}

// -----

module {
  func.func @missing_arg_attrs(%value: !torch.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'torchext.trident_kernel_launch' op expects arg_attrs for every kernel operand}}
    "torchext.trident_kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>}> : (i64, i64, i64, i64, i64, i64, !torch.int) -> ()
    func.return
  }
}

// -----

module {
  func.func @invalid_specialization_kind(%value: !torch.float) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'torchext.trident_kernel_launch' op kernel operand #0 of type '!torch.float' cannot be converted to specialization kind 'i32'}}
    "torchext.trident_kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{arg_attrs = [{triton.specialization = #torchext.specialization<kind = i32, divisibility = 1>}], kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>}> : (i64, i64, i64, i64, i64, i64, !torch.float) -> ()
    func.return
  }
}

// -----

module {
  func.func @missing_specialization(%value: !torch.int) {
    %one = arith.constant 1 : i64
    // expected-error@+1 {{'torchext.trident_kernel_launch' op kernel operand #0 requires a triton.specialization attribute}}
    "torchext.trident_kernel_launch"(%one, %one, %one, %one, %one, %one, %value) <{arg_attrs = [{}], kernel = @kernel::@entry, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1>}> : (i64, i64, i64, i64, i64, i64, !torch.int) -> ()
    func.return
  }
}
