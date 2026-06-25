// RUN: libtriton-core-opt %s --convert-to-llvm -split-input-file | FileCheck %s

// Globals and function declarations at module level (order may vary).
// CHECK-DAG:  llvm.mlir.global internal constant @__libtriton_constant_overload_
// CHECK-DAG:  llvm.mlir.global internal constant @"__libtriton_constant_op_aten::empty_like"
// CHECK-DAG:  llvm.func @aoti_torch_call_dispatcher
// CHECK-DAG:  llvm.func @mLibTritonTVMFFIObjectToTensor
// CHECK-DAG:  llvm.func @mLibTritonTensorToTVMFFIObject

// CHECK-LABEL: llvm.func @aten_empty_like(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:         llvm.alloca {{%.*}} x i64
// Unpack TVMFFIObjectHandle -> AtenTensorHandle.
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor(%[[ARG0]],
// CHECK:         llvm.ptrtoint {{%.*}} : !llvm.ptr to i64
// CHECK:         llvm.store {{%.*}}, {{%.*}} : i64, !llvm.ptr
// Call dispatcher.
// CHECK:         llvm.mlir.addressof @"__libtriton_constant_op_aten::empty_like"
// CHECK:         llvm.mlir.addressof @__libtriton_constant_overload_
// CHECK:         llvm.call @aoti_torch_call_dispatcher
// CHECK-SAME:      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// Load result from slot[0] and pack back.
// CHECK:         llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK:         llvm.inttoptr {{%.*}} : i64 to !llvm.ptr
// CHECK:         llvm.call @mLibTritonTensorToTVMFFIObject
// CHECK:         llvm.load
// CHECK:         llvm.return

module {
func.func @aten_empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return %0 : !torch.vtensor<[200,200,26],f64>
}
}

// -----

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


