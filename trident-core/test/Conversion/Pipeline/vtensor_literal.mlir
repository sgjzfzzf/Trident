// RUN: trident-core-opt %s --torch-to-llvm-pipeline | FileCheck %s
//
// This test verifies that lowering dispatches solely by dense.isSplat():
// - splat literal: aoti_torch_aten_full path
// - non-splat literal: CPU staging + aoti_torch_copy_ path

// CHECK-DAG: llvm.func @aoti_torch_aten_full(!llvm.ptr, i64, f64, {{.*}}) -> i32
// CHECK-DAG: llvm.func @aoti_torch_copy_(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-DAG: llvm.func @aoti_torch_empty_strided(i64, !llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @aoti_torch_create_tensor_from_blob(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @torch.vtensor.literal.splat() -> !llvm.struct<(i32, i32, i64)> {
// CHECK:         llvm.call @mTridentTVMFFIToTorchType({{.*}}) : (i8, i8) -> i32
// CHECK:         llvm.call @aoti_torch_aten_full({{.*}})
// CHECK:         llvm.call @mTridentTensorToTVMFFIObject
// CHECK:         llvm.return {{.*}} : !llvm.struct<(i32, i32, i64)>
func.func @torch.vtensor.literal.splat() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<1.250000e+00> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @torch.vtensor.literal.nonsplat() -> !llvm.struct<(i32, i32, i64)> {
// CHECK:         llvm.call @mTridentTVMFFIToTorchType({{.*}}) : (i8, i8) -> i32
// CHECK:         llvm.call @mTridentTVMFFIDeviceToTorchDeviceType({{.*}}) : (i32) -> i32
// CHECK:         llvm.call @aoti_torch_get_current_device_index({{.*}}) : (!llvm.ptr) -> i32
// CHECK:         llvm.call @aoti_torch_create_tensor_from_blob({{.*}})
// CHECK:         llvm.call @aoti_torch_empty_strided({{.*}})
// CHECK:         llvm.call @aoti_torch_copy_({{.*}})
// CHECK:         llvm.call @aoti_torch_delete_tensor_object({{.*}})
// CHECK:         llvm.call @mTridentTensorToTVMFFIObject
// CHECK:         llvm.return {{.*}} : !llvm.struct<(i32, i32, i64)>
func.func @torch.vtensor.literal.nonsplat() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_vtensor_literal_splat(
// The wrapper passes through the lowered TVMFFIAny result from the inner
// function and stores it into the return slot expected by tvm_ffi.func.
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.vtensor.literal.splat()
// CHECK:         llvm.store %[[CALLEE_RET]], {{.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @vtensor_literal_splat(%dummy: !torch.int) -> !torch.vtensor<[2,3],f32> {
  %0 = func.call @torch.vtensor.literal.splat() : () -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

tvm_ffi.func @vtensor_literal_nonsplat(%dummy: !torch.int) -> !torch.vtensor<[2,3],f32> {
  %0 = func.call @torch.vtensor.literal.nonsplat() : () -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}
