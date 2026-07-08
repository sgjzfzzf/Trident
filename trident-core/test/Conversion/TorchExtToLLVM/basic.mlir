//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s
//
// Tests that the standalone TorchExtToLLVM pass (invoked as part of the
// --trident-lowering-pipeline) lowers TorchExt dialect ops to LLVM.

// CHECK-DAG:   llvm.func @TVMFFIObjectIncRef(!llvm.ptr) -> i32
// CHECK-DAG:   llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @object_inc_ref(
// CHECK-SAME:    %[[OBJ:.*]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK-NEXT:    %[[DATA:.*]] = llvm.extractvalue %[[OBJ]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    %[[PTR:.*]] = llvm.inttoptr %[[DATA]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[CALL:.*]] = llvm.call @TVMFFIObjectIncRef(%[[PTR]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @object_inc_ref(%object: !torch.vtensor<[3,4],f32>) {
  torchext.ObjectIncRef %object : !torch.vtensor<[3,4],f32>
  return
}

// -----

// CHECK-LABEL: llvm.func @object_dec_ref(
// CHECK-SAME:    %[[OBJ:.*]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK-NEXT:    %[[DATA:.*]] = llvm.extractvalue %[[OBJ]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    %[[PTR:.*]] = llvm.inttoptr %[[DATA]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[CALL:.*]] = llvm.call @TVMFFIObjectDecRef(%[[PTR]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @object_dec_ref(%object: !torch.list<int>) {
  torchext.ObjectDecRef %object : !torch.list<int>
  return
}

