// RUN: libtriton-core-opt %s --convert-torch-to-aoti | FileCheck %s

// CHECK-LABEL:   func.func @torch.aten.empty_like(
// CHECK-SAME:                                %[[SELF:.*]]: !torch.vtensor<[200,200,26],f64>) {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK-NEXT:      %[[FALSE:.*]] = torch.constant.bool false
// CHECK:           %[[DISPATCH:.*]] = aoti.torch_call_dispatcher "aten::empty_like" : ""
// CHECK-SAME:         (%[[SELF]], %[[NONE]], %[[NONE]], %[[NONE]], %[[FALSE]], %[[NONE]])
// CHECK-SAME:         : (!torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[200,200,26],f64>
// CHECK-NEXT:      return
func.func @torch.aten.empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return
}
