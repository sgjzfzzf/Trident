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

// CHECK-DAG: llvm.func @aoti_torch_aten_full(!llvm.ptr, i64, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @aoti_torch_copy_(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @aoti_torch_empty_strided(i64, !llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @aoti_torch_create_tensor_from_blob(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @torch.vtensor.literal.splat() -> !llvm.struct<(i32, i32, i64)> {
// CHECK:         %[[FILL_VAL:.*]] = llvm.mlir.constant(1.250000e+00 : f64) : f64
// CHECK:         %[[ZERO_PTR:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         %[[DEV_CONST:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[NDIM:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:         %[[F32_DTYPE:.*]] = llvm.mlir.constant(2 : i8) : i8
// CHECK:         %[[F32_BITS:.*]] = llvm.mlir.constant(32 : i8) : i8
// CHECK:         %[[TORCH_DTYPE:.*]] = llvm.call @mTridentTVMFFIToTorchType(%[[F32_DTYPE]], %[[F32_BITS]]) : (i8, i8) -> i32
// CHECK:         %[[SHAPE_ALLOCA:.*]] = llvm.alloca %[[NDIM]] x i64 : (i64) -> !llvm.ptr
// CHECK:         %[[DEVICE_TYPE:.*]] = llvm.call @mTridentTVMFFIDeviceToTorchDeviceType(%[[DEV_CONST]]) : (i32) -> i32
// CHECK:         %[[DTYPE_SLOT:.*]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// CHECK:         %[[DEV_SLOT:.*]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// CHECK:         %[[DEV_IDX_SLOT:.*]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// CHECK:         %[[DEV_IDX_CALL:.*]] = llvm.call @aoti_torch_get_current_device_index(%[[DEV_IDX_SLOT]]) : (!llvm.ptr) -> i32
// CHECK:         %[[DEV_IDX:.*]] = llvm.load %[[DEV_IDX_SLOT]] : !llvm.ptr -> i32
// CHECK:         %[[TENSOR_SLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         %[[FULL_CALL:.*]] = llvm.call @aoti_torch_aten_full(%[[SHAPE_ALLOCA]], %[[NDIM]], %[[FILL_VAL]], %[[DTYPE_SLOT]], %[[ZERO_PTR]], %[[DEV_SLOT]], %[[DEV_IDX]], %[[ZERO_PTR]], %[[TENSOR_SLOT]]) : (!llvm.ptr, i64, f64, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[TENSOR_PTR:.*]] = llvm.load %[[TENSOR_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[OBJ_SLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         %[[TO_OBJ:.*]] = llvm.call @mTridentTensorToTVMFFIObject(%[[TENSOR_PTR]], %[[OBJ_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[OBJ_PTR:.*]] = llvm.load %[[OBJ_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[PTR_INT:.*]] = llvm.ptrtoint %[[OBJ_PTR]] : !llvm.ptr to i64
// CHECK:         llvm.return %[[RETVAL:.*]] : !llvm.struct<(i32, i32, i64)>
func.func @torch.vtensor.literal.splat() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<1.250000e+00> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @torch.vtensor.literal.nonsplat() -> !llvm.struct<(i32, i32, i64)> {
// CHECK:         %[[DEV_2_CONST:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         %[[DEV_1_CONST:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[ZERO_I32:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[ZERO_I64:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         %[[SIX:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK:         %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         %[[NDIM:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:         %[[F32_DTYPE:.*]] = llvm.mlir.constant(2 : i8) : i8
// CHECK:         %[[F32_BITS:.*]] = llvm.mlir.constant(32 : i8) : i8
// CHECK:         %[[TORCH_DTYPE:.*]] = llvm.call @mTridentTVMFFIToTorchType(%[[F32_DTYPE]], %[[F32_BITS]]) : (i8, i8) -> i32
// CHECK:         %[[SHAPE_ALLOCA:.*]] = llvm.alloca %[[NDIM]] x i64 : (i64) -> !llvm.ptr
// CHECK:         %[[STRIDES_ALLOCA:.*]] = llvm.alloca %[[NDIM]] x i64 : (i64) -> !llvm.ptr
// CHECK:         %[[DATA_ALLOCA:.*]] = llvm.alloca %[[SIX]] x i32 : (i64) -> !llvm.ptr
// CHECK:         %[[DEVICE_TYPE_CPU:.*]] = llvm.call @mTridentTVMFFIDeviceToTorchDeviceType(%[[DEV_1_CONST]]) : (i32) -> i32
// CHECK:         %[[DEVICE_TYPE_SCALAR:.*]] = llvm.call @mTridentTVMFFIDeviceToTorchDeviceType(%[[DEV_2_CONST]]) : (i32) -> i32
// CHECK:         %[[DEV_IDX_SLOT:.*]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// CHECK:         %[[DEV_IDX_CALL:.*]] = llvm.call @aoti_torch_get_current_device_index(%[[DEV_IDX_SLOT]]) : (!llvm.ptr) -> i32
// CHECK:         %[[DEV_IDX:.*]] = llvm.load %[[DEV_IDX_SLOT]] : !llvm.ptr -> i32
// CHECK:         %[[BLOB_SLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         %[[FROM_BLOB:.*]] = llvm.call @aoti_torch_create_tensor_from_blob(%[[DATA_ALLOCA]], %[[NDIM]], %[[SHAPE_ALLOCA]], %[[STRIDES_ALLOCA]], %[[ZERO_I64]], %[[TORCH_DTYPE]], %[[DEVICE_TYPE_CPU]], %[[ZERO_I32]], %[[BLOB_SLOT]]) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         %[[BLOB_TENSOR:.*]] = llvm.load %[[BLOB_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[STRIDED_SLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         %[[EMPTY_STRIDED:.*]] = llvm.call @aoti_torch_empty_strided(%[[NDIM]], %[[SHAPE_ALLOCA]], %[[STRIDES_ALLOCA]], %[[TORCH_DTYPE]], %[[DEVICE_TYPE_SCALAR]], %[[DEV_IDX]], %[[STRIDED_SLOT]]) : (i64, !llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
// CHECK:         %[[STRIDED_TENSOR:.*]] = llvm.load %[[STRIDED_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[COPY:.*]] = llvm.call @aoti_torch_copy_(%[[STRIDED_TENSOR]], %[[BLOB_TENSOR]], %[[ZERO_I32]]) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:         %[[DELETE:.*]] = llvm.call @aoti_torch_delete_tensor_object(%[[BLOB_TENSOR]]) : (!llvm.ptr) -> i32
// CHECK:         %[[OBJ_SLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         %[[TO_OBJ:.*]] = llvm.call @mTridentTensorToTVMFFIObject(%[[STRIDED_TENSOR]], %[[OBJ_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         %[[OBJ_PTR:.*]] = llvm.load %[[OBJ_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[PTR_INT:.*]] = llvm.ptrtoint %[[OBJ_PTR]] : !llvm.ptr to i64
// CHECK:         llvm.return %[[RETVAL:.*]] : !llvm.struct<(i32, i32, i64)>
func.func @torch.vtensor.literal.nonsplat() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_vtensor_literal_splat(
// The wrapper passes through the lowered TVMFFIAny result from the inner
// function and stores it into the return slot expected by tvm_ffi.func.
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.vtensor.literal.splat()
// CHECK:         llvm.store %[[CALLEE_RET]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @vtensor_literal_splat(%dummy: !torch.int) -> !torch.vtensor<[2,3],f32> {
  %0 = func.call @torch.vtensor.literal.splat() : () -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_vtensor_literal_nonsplat(
// The wrapper passes through the lowered TVMFFIAny result from the inner
// function and stores it into the return slot expected by tvm_ffi.func.
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.vtensor.literal.nonsplat()
// CHECK:         llvm.store %[[CALLEE_RET]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @vtensor_literal_nonsplat(%dummy: !torch.int) -> !torch.vtensor<[2,3],f32> {
  %0 = func.call @torch.vtensor.literal.nonsplat() : () -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}
