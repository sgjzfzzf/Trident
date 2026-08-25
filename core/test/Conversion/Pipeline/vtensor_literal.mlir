//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that lowering dispatches solely by dense.isSplat():
// - splat literal: aoti_torch_aten_full path
// - non-splat literal: CPU staging + aoti_torch_copy_ path

// CHECK-LABEL: llvm.func @torch.vtensor.literal.splat(
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: %[[SHAPE_DATA:.*]] = llvm.alloca %[[SHAPE_COUNT:.*]] x i64
// CHECK: %[[LITERAL_DATA:.*]] = llvm.alloca %[[LITERAL_COUNT:.*]] x f32
// CHECK: llvm.call @aoti_torch_aten_full
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK-LABEL: llvm.func @torch.vtensor.literal.nonsplat(
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: llvm.call @aoti_torch_create_tensor_from_blob
// CHECK: llvm.call @aoti_torch_empty_strided
// CHECK: llvm.call @aoti_torch_copy_
// CHECK: llvm.call @aoti_torch_delete_tensor_object
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK-LABEL: llvm.func @__tvm_ffi_vtensor_literal_splat(
// CHECK: llvm.call @vtensor_literal_splat
// CHECK-LABEL: llvm.func @__tvm_ffi_vtensor_literal_nonsplat(
// CHECK: llvm.call @vtensor_literal_nonsplat

func.func @torch.vtensor.literal.splat() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<1.250000e+00> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

func.func @torch.vtensor.literal.nonsplat() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// The wrapper passes through the lowered TVMFFIAny result from the inner
// function and stores it into the return slot expected by tvm_ffi.func.
tvm_ffi.func @vtensor_literal_splat() -> !torch.vtensor<[2,3],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.vtensor.literal(dense<1.250000e+00> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

// The wrapper passes through the lowered TVMFFIAny result from the inner
// function and stores it into the return slot expected by tvm_ffi.func.
tvm_ffi.func @vtensor_literal_nonsplat() -> !torch.vtensor<[2,3],f32> attributes {emit_tvm_ffi_abi} {
  %0 = torch.vtensor.literal(dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}
