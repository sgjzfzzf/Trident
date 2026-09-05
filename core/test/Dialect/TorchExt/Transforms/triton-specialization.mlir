//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torch-to-tvm-ffi --convert-tvm-ffi-to-func --inline --decompose-specialization | FileCheck %s --check-prefix=CHECK-DECOMPOSE
// RUN: trident-core-opt %s --convert-torch-to-tvm-ffi --convert-tvm-ffi-to-func --inline --decompose-specialization | FileCheck %s --check-prefix=CHECK-INLINE

// CHECK-DECOMPOSE-LABEL: func.func @validate
// CHECK-DECOMPOSE:         torchext.trident_kernel_launch{{.*}}args(%[[TENSOR:[a-zA-Z0-9_]+]] : !tvm_ffi.tensor {test.keep = "yes"}, %[[VALUE:[a-zA-Z0-9_]+]] : !tvm_ffi.int)
// CHECK-DECOMPOSE:         return %{{.*}} : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
// CHECK-DECOMPOSE-NOT:     triton.specialization
module {
  func.func @validate(
      %tensor: !torch.vtensor<[4],f32>, %value: !torch.int)
      -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception> {
    %one = arith.constant 1 : i64
    torchext.trident_kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%tensor : !torch.vtensor<[4],f32> {test.keep = "yes", triton.specialization = #torchext.specialization<divisibility = 16>}, %value : !torch.int {triton.specialization = #torchext.specialization<divisibility = 16>})
    %success = tvm_ffi.cast %value : !torch.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
    func.return %success : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
  }

// CHECK-INLINE-LABEL: func.func @wrapper
// CHECK-INLINE-NOT:   call @main
// CHECK-INLINE:       %[[CHECK:.*]] = llvm.icmp "eq"
// CHECK-INLINE:       cf.cond_br %{{.*}}, ^[[LAUNCH:.*]], ^[[FAIL:.*]]
// CHECK-INLINE:     ^[[LAUNCH]]:
// CHECK-INLINE:       torchext.trident_kernel_launch
// CHECK-INLINE:       return %{{.*}} : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
// CHECK-INLINE:     ^[[FAIL]]:
// CHECK-INLINE:       %[[EXCEPTION:.*]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK-INLINE:       %[[FAILURE:.*]] = tvm_ffi.cast %[[EXCEPTION]] : !tvm_ffi.exception -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
// CHECK-INLINE:       return %[[FAILURE]] : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>

  func.func private @main(
      %tensor: !torch.vtensor<[4],f32>, %value: !torch.int) -> !torch.int {
    %one = arith.constant 1 : i64
    torchext.trident_kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%tensor : !torch.vtensor<[4],f32> {triton.specialization = #torchext.specialization<divisibility = 16>}, %value : !torch.int {triton.specialization = #torchext.specialization<divisibility = 16>})
    func.return %value : !torch.int
  }

tvm_ffi.func @wrapper(
    %tensor: !torch.vtensor<[4],f32>, %value: !torch.int)
    -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception> {
  %result = func.call @main(%tensor, %value)
      : (!torch.vtensor<[4],f32>, !torch.int) -> !torch.int
  %wrapped = tvm_ffi.cast %result
      : !torch.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
  tvm_ffi.return %wrapped
      : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
}
}
