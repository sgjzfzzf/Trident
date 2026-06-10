// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @call_empty_like
func.func @call_empty_like(%arg0: !torch.vtensor<[4],f32>) {
  // CHECK: %[[V:.*]] = torchext.aoti.CallDispatcher "aten::empty_like" : ""(%arg0) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
  %0 = torchext.aoti.CallDispatcher "aten::empty_like" : ""(%arg0) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
  return
}

// -----

// CHECK-LABEL: func.func @list_delete_list
func.func @list_delete_list(%list: !torch.list<int>) {
  // CHECK: torchext.aoti.ListDeleteList %arg0 : <int>
  torchext.aoti.ListDeleteList %list : !torch.list<int>
  return
}
