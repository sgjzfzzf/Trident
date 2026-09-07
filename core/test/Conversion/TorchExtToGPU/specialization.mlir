//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file --inline --convert-torchext-to-gpu | FileCheck %s

// CHECK-LABEL: tvm_ffi.func @validate
// CHECK-SAME:    %[[TENSOR:[a-zA-Z0-9_]+]]: !torch.vtensor
// CHECK-SAME:    %[[SPECIALIZED:[a-zA-Z0-9_]+]]: !torch.int
// CHECK-NOT:     call @launch
// CHECK:         %[[GUARD_OBJECT:[a-zA-Z0-9_]+]] = torchext.get %[[TENSOR]]
// CHECK:         %[[GUARD_TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[GUARD_OBJECT]]
// CHECK:         %[[GUARD_DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[GUARD_TENSOR]]
// CHECK:         %[[GUARD_CONSTANT:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[SPECIALIZED]]
// CHECK:         %[[FLOAT_VALUE:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[FLOAT:[a-zA-Z0-9_]+]]
// CHECK:         llvm.ptrtoint %[[GUARD_DATA]]
// CHECK:         llvm.urem
// CHECK:         %[[TENSOR_CHECKS:[a-zA-Z0-9_]+]] = arith.andi
// CHECK:         %[[EXPECTED:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[CONSTANT_CHECK:[a-zA-Z0-9_]+]] = llvm.icmp "eq" %[[GUARD_CONSTANT]], %[[EXPECTED]]
// CHECK:         %[[CHECKS_WITH_CONSTANT:[a-zA-Z0-9_]+]] = arith.andi %[[TENSOR_CHECKS]], %[[CONSTANT_CHECK]]
// CHECK:         %[[CAST:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast
// CHECK:         llvm.urem
// CHECK:         llvm.icmp "eq"
// CHECK:         %[[ALL_CHECKS:[a-zA-Z0-9_]+]] = arith.andi
// CHECK:         cf.cond_br %[[ALL_CHECKS]], [[SUCCESS:\^bb[0-9]+]], [[FAILURE:\^bb[0-9]+]]
// CHECK:       [[SUCCESS]]:
// CHECK:         %[[LAUNCH_OBJECT:[a-zA-Z0-9_]+]] = torchext.get %[[TENSOR]]
// CHECK:         %[[LAUNCH_TENSOR:[a-zA-Z0-9_]+]] = tvm_ffi.as %[[LAUNCH_OBJECT]]
// CHECK:         %[[LAUNCH_DATA:[a-zA-Z0-9_]+]] = dlpack.tensor.data %[[LAUNCH_TENSOR]]
// CHECK:         gpu.launch_func
// CHECK-SAME:    args(%[[LAUNCH_DATA]] : !llvm.ptr
// CHECK:         tvm_ffi.return
// CHECK:       [[FAILURE]]:
// CHECK:         tvm_ffi.exception "GuardMatch"
// CHECK:         tvm_ffi.return
// CHECK-NOT:     torchext.trident_kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func private @launch(%tensor: !torch.vtensor<[4],f32>,
      %specialized: !torch.int, %value: !torch.float)
      -> !tvm_ffi.int {
    %one = arith.constant 1 : i64
    torchext.trident_kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%tensor : !torch.vtensor<[4],f32> {triton.specialization = #torchext.variable_specialization<kind = !llvm.ptr, divisibility = 16>}, %specialized : !torch.int {triton.specialization = #torchext.constant_specialization<value = 1 : i64>}, %value : !torch.float {triton.specialization = #torchext.variable_specialization<kind = f32, divisibility = 16>})
    %result = tvm_ffi.constant.int 0
    return %result : !tvm_ffi.int
  }

  tvm_ffi.func @validate(%tensor: !torch.vtensor<[4],f32>, %value: !torch.float,
      %specialized: !torch.int)
      -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception> {
    %result = func.call @launch(%tensor, %specialized, %value)
        : (!torch.vtensor<[4],f32>, !torch.int, !torch.float) -> !tvm_ffi.int
    %success = tvm_ffi.cast %result : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
    tvm_ffi.return %success : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
  }
}

// -----

// CHECK-LABEL: tvm_ffi.func @validate_tuple
// CHECK-SAME:    %[[SHAPE:[a-zA-Z0-9_]+]]: !torch.tuple<int, int>
// CHECK:         tvm_ffi.array.create
// CHECK:         tvm_ffi.eq %[[SHAPE]]
// CHECK:         gpu.launch_func
// CHECK-NOT:     torchext.trident_kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  tvm_ffi.func @validate_tuple(%shape: !torch.tuple<int, int>)
      -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception> {
    %one = arith.constant 1 : i64
    torchext.trident_kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%shape : !torch.tuple<int, int> {triton.specialization = #torchext.constant_specialization<value = [2, 4]>})
    %result = tvm_ffi.constant.int 0
    %success = tvm_ffi.cast %result : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
    tvm_ffi.return %success : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
  }
}

// -----

// CHECK-LABEL: tvm_ffi.func @validate_string
// CHECK-SAME:    %[[ACTIVATION:[a-zA-Z0-9_]+]]: !torch.str
// CHECK:         cf.cond_br %[[INITIAL_STRING:[a-zA-Z0-9_]+]], [[SUCCESS_STRING:\^bb[0-9]+]], [[FAILURE_STRING:\^bb[0-9]+]]
// CHECK:       [[SUCCESS_STRING]]:
// CHECK:         gpu.launch_func
// CHECK-SAME:    args(%[[ZERO_STRING:[a-zA-Z0-9_]+]] : i64, %[[ZERO_STRING]] : i64)
// CHECK-NOT:     torch_c.to_
// CHECK-NOT:     llvm.icmp
// CHECK-NOT:     torchext.trident_kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func private @launch_string(%activation: !torch.str)
      -> !tvm_ffi.int {
    %one = arith.constant 1 : i64
    torchext.trident_kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%activation : !torch.str {triton.specialization = #torchext.constant_specialization<value = "leaky_relu">})
    %result = tvm_ffi.constant.int 0
    return %result : !tvm_ffi.int
  }

  tvm_ffi.func @validate_string(%activation: !torch.str)
      -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception> {
    %result = func.call @launch_string(%activation)
        : (!torch.str) -> !tvm_ffi.int
    %success = tvm_ffi.cast %result : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
    tvm_ffi.return %success : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
  }
}

// -----

// CHECK-LABEL: tvm_ffi.func @validate_scalar_constants
// CHECK-SAME:    %[[FLAG:[a-zA-Z0-9_]+]]: !torch.bool
// CHECK-SAME:    %[[VALUE:[a-zA-Z0-9_]+]]: !torch.int
// CHECK-SAME:    %[[SCALE:[a-zA-Z0-9_]+]]: !torch.float
// CHECK:         %[[FLAG_NATIVE:[a-zA-Z0-9_]+]] = torch_c.to_i1 %[[FLAG]]
// CHECK:         %[[VALUE_NATIVE:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[VALUE]]
// CHECK:         %[[SCALE_NATIVE:[a-zA-Z0-9_]+]] = torch_c.to_f64 %[[SCALE]]
// CHECK:         %[[INITIAL:[a-zA-Z0-9_]+]] = llvm.mlir.constant(true) : i1
// CHECK:         %[[TRUE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(true) : i1
// CHECK:         llvm.icmp "eq" %[[FLAG_NATIVE]], %[[TRUE]] : i1
// CHECK:         %[[EXPECTED_FLOAT:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1.000000e+00 : f64) : f64
// CHECK:         %[[SCALE_BITS:[a-zA-Z0-9_]+]] = llvm.bitcast %[[SCALE_NATIVE]] : f64 to i64
// CHECK:         %[[EXPECTED_BITS:[a-zA-Z0-9_]+]] = llvm.bitcast %[[EXPECTED_FLOAT]] : f64 to i64
// CHECK:         llvm.icmp "eq" %[[SCALE_BITS]], %[[EXPECTED_BITS]] : i64
// CHECK:         cf.cond_br
// CHECK:         %[[LAUNCH_VALUE:[a-zA-Z0-9_]+]] = torch_c.to_i64 %[[VALUE]]
// CHECK:         %[[LAUNCH_I32:[a-zA-Z0-9_]+]] = llvm.trunc %[[LAUNCH_VALUE]] : i64 to i32
// CHECK:         gpu.launch_func
// CHECK-SAME:    args(%[[LAUNCH_I32]] : i32
// CHECK-NOT:     torchext.trident_kernel_launch

module attributes {gpu.container_module} {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func private @launch_scalar_constants(%flag: !torch.bool,
      %value: !torch.int, %scale: !torch.float) -> !tvm_ffi.int {
    %one = arith.constant 1 : i64
    torchext.trident_kernel_launch @kernel::@entry
        blocks in (%one, %one, %one) : i64
        threads in (%one, %one, %one)
        args (%flag : !torch.bool {triton.specialization = #torchext.constant_specialization<value = true>}, %value : !torch.int {triton.specialization = #torchext.variable_specialization<kind = i32>}, %scale : !torch.float {triton.specialization = #torchext.constant_specialization<value = 1.000000e+00 : f64>})
    %result = tvm_ffi.constant.int 0
    return %result : !tvm_ffi.int
  }

  tvm_ffi.func @validate_scalar_constants(%flag: !torch.bool,
      %value: !torch.int, %scale: !torch.float)
      -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception> {
    %result = func.call @launch_scalar_constants(%flag, %value, %scale)
        : (!torch.bool, !torch.int, !torch.float) -> !tvm_ffi.int
    %success = tvm_ffi.cast %result : !tvm_ffi.int -> !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
    tvm_ffi.return %success : !tvm_ffi.union<!tvm_ffi.int, !tvm_ffi.exception>
  }
}
