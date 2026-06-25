// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @list_delete_list
func.func @list_delete_list(%list: !torch.list<int>) {
  // CHECK: torchext.aoti.ListDeleteList %arg0 : <int>
  torchext.aoti.ListDeleteList %list : !torch.list<int>
  return
}
