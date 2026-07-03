//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --torch-to-llvm-pipeline | FileCheck %s
//
// Tests that the standalone TorchExtToLLVM pass (invoked as part of the
// --torch-to-llvm-pipeline) lowers TorchExt dialect ops to LLVM.

// CHECK-DAG:   llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @object_dec_ref(
// CHECK-SAME:    %[[OBJ:.*]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK-NEXT:    llvm.extractvalue %[[OBJ]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.inttoptr {{%.*}} : i64 to !llvm.ptr
// CHECK:         llvm.call @TVMFFIObjectDecRef({{%.*}}) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return

module {
func.func @object_dec_ref(%object: !torch.list<int>) {
  torchext.ObjectDecRef %object : !torch.list<int>
  return
}
}
