//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// Tensor literals use one stable CPU-staging path for both splat and
// non-splat attributes before copying the tensor to the current CUDA device.

// CHECK-LABEL: llvm.func @torch.vtensor.literal.splat(
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[SPLAT_NAME:[a-zA-Z0-9_]+]], %[[SPLAT_HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[SPLAT_HANDLE:[a-zA-Z0-9_]+]], %[[SPLAT_RESULT_SLOT:[a-zA-Z0-9_]+]], %[[SPLAT_NARGS:[a-zA-Z0-9_]+]], %[[SPLAT_ARGS:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: %[[SHAPE_DATA:[a-zA-Z0-9_]+]] = llvm.alloca %[[SHAPE_COUNT:[a-zA-Z0-9_]+]] x i64
// CHECK: %[[LITERAL_DATA:[a-zA-Z0-9_]+]] = llvm.alloca %[[LITERAL_COUNT:[a-zA-Z0-9_]+]] x f32
// CHECK: llvm.call @aoti_torch_create_tensor_from_blob
// CHECK: llvm.call @aoti_torch_empty_strided
// CHECK: llvm.call @aoti_torch_copy_
// CHECK: llvm.call @aoti_torch_delete_tensor_object
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[NONSPLAT_GLOBAL_NAME:[a-zA-Z0-9_]+]], %[[NONSPLAT_HANDLE_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[NONSPLAT_HANDLE:[a-zA-Z0-9_]+]], %[[NONSPLAT_RESULT_SLOT:[a-zA-Z0-9_]+]], %[[NONSPLAT_NARGS:[a-zA-Z0-9_]+]], %[[NONSPLAT_ARGS:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-LABEL: llvm.func @torch.vtensor.literal.nonsplat(
// CHECK: llvm.call @TVMFFIFunctionGetGlobal(%[[NONSPLAT_NAME:[a-zA-Z0-9_]+]], %[[NONSPLAT_GLOBAL_SLOT:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @TVMFFIFunctionCall(%[[NONSPLAT_FUNCTION:[a-zA-Z0-9_]+]], %[[NONSPLAT_RESULT:[a-zA-Z0-9_]+]], %[[NONSPLAT_COUNT:[a-zA-Z0-9_]+]], %[[NONSPLAT_ARGUMENTS:[a-zA-Z0-9_]+]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK: llvm.call @aoti_torch_create_tensor_from_blob
// CHECK: llvm.call @aoti_torch_empty_strided
// CHECK: llvm.call @aoti_torch_copy_
// CHECK: llvm.call @aoti_torch_delete_tensor_object
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK-LABEL: llvm.func @__tvm_ffi_vtensor_literal_splat(
// CHECK: llvm.call @vtensor_literal_splat() : () -> !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_vtensor_literal_nonsplat(
// CHECK: llvm.call @vtensor_literal_nonsplat() : () -> !llvm.struct<(i32, i32, i64)>

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
