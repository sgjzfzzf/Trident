//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// This test verifies that sub NumberType shims use symbols exported by AOTI:
//   - aten.sub.Scalar -> aoti_torch_cuda_add_Scalar
//   - aten.sub.Tensor -> aoti_torch_aten_subtract_Tensor

// CHECK-DAG: llvm.func @aoti_torch_cuda_add_Scalar(!llvm.ptr, f64, f64, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @aoti_torch_aten_subtract_Tensor(!llvm.ptr, !llvm.ptr, f64, !llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @torch.aten.sub.Scalar(
// CHECK-SAME:    %[[TENSOR:.*]]: !llvm.struct<(i32, i32, i64)>, %[[ALPHA:.*]]: !llvm.struct<(i32, i32, i64)>, %[[SCALAR:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK:         llvm.extractvalue %[[TENSOR]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.call @mTridentTVMFFIObjectToTensor({{%.*}},
// CHECK:         %[[ALPHA_TYPEIDX:.*]] = llvm.extractvalue %[[ALPHA]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[ALPHA_PAYLOAD:.*]] = llvm.extractvalue %[[ALPHA]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[ALPHA_IS_INT:.*]] = llvm.icmp "eq" %[[ALPHA_TYPEIDX]], {{%.*}} : i32
// CHECK:         %[[ALPHA_INT_AS_F64:.*]] = llvm.sitofp %[[ALPHA_PAYLOAD]] : i64 to f64
// CHECK:         %[[ALPHA_FLOAT_AS_F64:.*]] = llvm.bitcast %[[ALPHA_PAYLOAD]] : i64 to f64
// CHECK:         %[[ALPHA_VALUE:.*]] = llvm.select %[[ALPHA_IS_INT]], %[[ALPHA_INT_AS_F64]], %[[ALPHA_FLOAT_AS_F64]] : i1, f64
// CHECK:         %[[NEG_ALPHA:.*]] = llvm.fneg %[[ALPHA_VALUE]] : f64
// CHECK:         %[[SCALAR_TYPEIDX:.*]] = llvm.extractvalue %[[SCALAR]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_PAYLOAD:.*]] = llvm.extractvalue %[[SCALAR]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_IS_INT:.*]] = llvm.icmp "eq" %[[SCALAR_TYPEIDX]], {{%.*}} : i32
// CHECK:         %[[SCALAR_INT_AS_F64:.*]] = llvm.sitofp %[[SCALAR_PAYLOAD]] : i64 to f64
// CHECK:         %[[SCALAR_FLOAT_AS_F64:.*]] = llvm.bitcast %[[SCALAR_PAYLOAD]] : i64 to f64
// CHECK:         %[[SCALAR_VALUE:.*]] = llvm.select %[[SCALAR_IS_INT]], %[[SCALAR_INT_AS_F64]], %[[SCALAR_FLOAT_AS_F64]] : i1, f64
// CHECK:         llvm.call @aoti_torch_cuda_add_Scalar({{%.*}}, %[[NEG_ALPHA]], %[[SCALAR_VALUE]], {{%.*}})
// CHECK:         llvm.call @mTridentTensorToTVMFFIObject({{%.*}},
// CHECK:         llvm.return {{%.*}} : !llvm.struct<(i32, i32, i64)>
func.func @torch.aten.sub.Scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.sub.Scalar %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.float, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @torch.aten.sub.Tensor(
// CHECK-SAME:    %[[LHS:.*]]: !llvm.struct<(i32, i32, i64)>, %[[RHS:.*]]: !llvm.struct<(i32, i32, i64)>, %[[ALPHA:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK:         llvm.extractvalue %[[LHS]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.call @mTridentTVMFFIObjectToTensor({{%.*}},
// CHECK:         llvm.extractvalue %[[RHS]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.call @mTridentTVMFFIObjectToTensor({{%.*}},
// CHECK:         %[[ALPHA_TYPEIDX:.*]] = llvm.extractvalue %[[ALPHA]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[ALPHA_PAYLOAD:.*]] = llvm.extractvalue %[[ALPHA]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[ALPHA_IS_INT:.*]] = llvm.icmp "eq" %[[ALPHA_TYPEIDX]], {{%.*}} : i32
// CHECK:         %[[ALPHA_INT_AS_F64:.*]] = llvm.sitofp %[[ALPHA_PAYLOAD]] : i64 to f64
// CHECK:         %[[ALPHA_FLOAT_AS_F64:.*]] = llvm.bitcast %[[ALPHA_PAYLOAD]] : i64 to f64
// CHECK:         %[[ALPHA_VALUE:.*]] = llvm.select %[[ALPHA_IS_INT]], %[[ALPHA_INT_AS_F64]], %[[ALPHA_FLOAT_AS_F64]] : i1, f64
// CHECK:         llvm.call @aoti_torch_aten_subtract_Tensor({{%.*}}, %[[ALPHA_VALUE]], {{%.*}})
// CHECK:         llvm.call @mTridentTensorToTVMFFIObject({{%.*}},
// CHECK:         llvm.return {{%.*}} : !llvm.struct<(i32, i32, i64)>
func.func @torch.aten.sub.Tensor(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.float -> !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_sub_scalar(
// CHECK-SAME:    %arg0: !llvm.ptr, %[[ARGS:.*]]: !llvm.ptr, %arg2: i32, %[[RET:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[TENSOR_LOAD:.*]] = llvm.load %[[ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[ALPHA_GEP:.*]] = llvm.getelementptr %[[ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[ALPHA_LOAD:.*]] = llvm.load %[[ALPHA_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_GEP:.*]] = llvm.getelementptr %[[ARGS]][2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[SCALAR_LOAD:.*]] = llvm.load %[[SCALAR_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.aten.sub.Scalar(%[[TENSOR_LOAD]], %[[ALPHA_LOAD]], %[[SCALAR_LOAD]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.store %[[CALLEE_RET]], %[[RET]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @sub_scalar(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.float, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = func.call @torch.aten.sub.Scalar(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3],f32>, !torch.float, !torch.float) -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: llvm.func @__tvm_ffi_sub_tensor(
// CHECK-SAME:    %arg0: !llvm.ptr, %[[ARGS:.*]]: !llvm.ptr, %arg2: i32, %[[RET:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[LHS_LOAD:.*]] = llvm.load %[[ARGS]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[RHS_GEP:.*]] = llvm.getelementptr %[[ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[RHS_LOAD:.*]] = llvm.load %[[RHS_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[ALPHA_GEP:.*]] = llvm.getelementptr %[[ARGS]][2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[ALPHA_LOAD:.*]] = llvm.load %[[ALPHA_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:         %[[CALLEE_RET:.*]] = llvm.call @torch.aten.sub.Tensor(%[[LHS_LOAD]], %[[RHS_LOAD]], %[[ALPHA_LOAD]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.store %[[CALLEE_RET]], %[[RET]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @sub_tensor(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>, %arg2: !torch.float) -> !torch.vtensor<[2,3],f32> {
  %0 = func.call @torch.aten.sub.Tensor(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.float) -> !torch.vtensor<[2,3],f32>
  tvm_ffi.return %0 : !torch.vtensor<[2,3],f32>
}
