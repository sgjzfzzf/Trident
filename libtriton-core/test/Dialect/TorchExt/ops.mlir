// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @call_empty_like
func.func @call_empty_like(%arg0: !torch.vtensor<[4],f32>) {
  // CHECK: %[[V:.*]] = torchext.call_dispatcher "aten::empty_like" : ""(%arg0) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
  %0 = torchext.call_dispatcher "aten::empty_like" : ""(%arg0) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
  return
}
