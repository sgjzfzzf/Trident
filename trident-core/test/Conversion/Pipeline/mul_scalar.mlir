//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that torch.aten.mul.Scalar lowers to the dedicated AOTI
// entry point and still exercises NumberType scalar adaptation.

// CHECK:       llvm.func @aoti_torch_cuda_mul_Scalar(!llvm.ptr, f64, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @torch.aten.mul.Scalar(
// CHECK-SAME:    %[[TENSOR:.*]]: !llvm.struct<(i32, i32, i64)>, %[[SCALAR:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK:         llvm.extractvalue %[[TENSOR]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_TYPEIDX:.*]] = llvm.extractvalue %[[SCALAR]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_PAYLOAD:.*]] = llvm.extractvalue %[[SCALAR]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_IS_INT:.*]] = llvm.icmp "eq" %[[SCALAR_TYPEIDX]], {{%.*}} : i32
// CHECK:         %[[SCALAR_INT_AS_F64:.*]] = llvm.sitofp %[[SCALAR_PAYLOAD]] : i64 to f64
// CHECK:         %[[SCALAR_FLOAT_AS_F64:.*]] = llvm.bitcast %[[SCALAR_PAYLOAD]] : i64 to f64
// CHECK:         %[[SCALAR_VALUE:.*]] = llvm.select %[[SCALAR_IS_INT]], %[[SCALAR_INT_AS_F64]], %[[SCALAR_FLOAT_AS_F64]] : i1, f64
// CHECK:         llvm.call @aoti_torch_cuda_mul_Scalar({{%.*}})
// CHECK:         llvm.call @mTridentTensorToTVMFFIObject({{%.*}},
// CHECK:         llvm.return {{%.*}} : !llvm.struct<(i32, i32, i64)>
func.func @torch.aten.mul.Scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.mul.Scalar %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_mul_scalar(
// CHECK-SAME:    %arg0: !llvm.ptr, %[[ARGS:.*]]: !llvm.ptr, %arg2: i32, %[[RET:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[TENSOR_LOAD:.*]] = llvm.load %[[ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_GEP:.*]] = llvm.getelementptr %[[ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_LOAD:.*]] = llvm.load %[[SCALAR_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.aten.mul.Scalar(%[[TENSOR_LOAD]], %[[SCALAR_LOAD]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.store %[[CALLEE_RET]], %[[RET]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @mul_scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = func.call @torch.aten.mul.Scalar(%arg0, %arg1) : (!torch.vtensor<[2,3],f32>, !torch.float) -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}