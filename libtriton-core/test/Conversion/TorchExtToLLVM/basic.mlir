// RUN: libtriton-core-opt %s --convert-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @list_construct(
// CHECK-SAME:    %[[A:.*]]: i64, %[[B:.*]]: i64) -> !llvm.ptr {
// CHECK:      %[[HL_PTR:.*]] = llvm.alloca %{{.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT: %[[N:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: llvm.call @torch_new_list_reserve_size(%[[N]], %[[HL_PTR]]) : (i64, !llvm.ptr) -> i32
// CHECK-NEXT: %[[HL:.*]] = llvm.load %[[HL_PTR]] : !llvm.ptr -> !llvm.ptr

// CHECK:      llvm.call @torch_list_push_back(%[[HL]], %[[A]]) : (!llvm.ptr, i64) -> i32
// CHECK-NEXT: llvm.call @torch_list_push_back(%[[HL]], %[[B]]) : (!llvm.ptr, i64) -> i32

// CHECK-NEXT: llvm.return %[[HL]] : !llvm.ptr

module {
func.func @list_construct(%arg0: !torch.int, %arg1: !torch.int) -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  return %0 : !torch.list<int>
}
}

// CHECK-LABEL: llvm.func @list_delete_list(
// CHECK-SAME:    %[[LIST:.*]]: !llvm.ptr) {
// CHECK:      llvm.call @torch_delete_list(%[[LIST]]) : (!llvm.ptr) -> i32
// CHECK-NEXT: llvm.return

module {
func.func @list_delete_list(%list: !torch.list<int>) {
  torchext.aoti.ListDeleteList %list : !torch.list<int>
  return
}
}


