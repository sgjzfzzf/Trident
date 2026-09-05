//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torchext-to-gpu | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @validate
// CHECK:         llvm.ptrtoint
// CHECK:         %[[CAST:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[FLOAT:[a-zA-Z0-9_]+]] : f32 to i64
// CHECK:         llvm.urem
// CHECK:         llvm.icmp "eq"
// CHECK:         %[[ALL_CHECKS:[a-zA-Z0-9_]+]] = arith.andi
// CHECK:         cf.cond_br %[[ALL_CHECKS]], [[SUCCESS:\^bb[0-9]+]], [[FAILURE:\^bb[0-9]+]]
// CHECK:       [[SUCCESS]]:
// CHECK:         gpu.launch_func
// CHECK:         tvm_ffi.return
// CHECK:       [[FAILURE]]:
// CHECK:         tvm_ffi.exception "GuardMatch"
// CHECK:         tvm_ffi.return
// CHECK-NOT:     torchext.trident_kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  tvm_ffi.func @validate(%tensor: !torch.vtensor<[4],f32>, %value: !torch.float)
      -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception> {
    %one = arith.constant 1 : i64
    torchext.trident_kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%tensor : !torch.vtensor<[4],f32> {triton.specialization = #torchext.specialization<kind = !llvm.ptr, divisibility = 16>}, %value : !torch.float {triton.specialization = #torchext.specialization<kind = f32, divisibility = 16>})
    %result = tvm_ffi.constant.int 0
    %success = tvm_ffi.cast %result : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
    tvm_ffi.return %success : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
  }
}
