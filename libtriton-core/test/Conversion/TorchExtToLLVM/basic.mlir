// RUN: libtriton-core-opt %s --torch-to-llvm-pipeline | FileCheck %s
//
// Tests that the standalone TorchExtToLLVM pass (invoked as part of the
// --torch-to-llvm-pipeline) lowers TorchExt dialect ops to LLVM.

// CHECK-DAG:   llvm.func @TVMFFIObjectDecRef(!llvm.ptr) -> i32

// CHECK-LABEL: llvm.func @list_delete_list(
// CHECK-SAME:    %[[LIST:.*]]: !llvm.struct<(i32, i32, i64)>) {
// CHECK-NEXT:    llvm.extractvalue %[[LIST]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:         llvm.inttoptr {{%.*}} : i64 to !llvm.ptr
// CHECK:         llvm.call @TVMFFIObjectDecRef({{%.*}}) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return

module {
func.func @list_delete_list(%list: !torch.list<int>) {
  torchext.aoti.ListDeleteList %list : !torch.list<int>
  return
}
}
