// RUN: libtriton-core-opt %s --libtriton-convert-torch-to-arith | FileCheck %s

// CHECK-LABEL:   func.func @torch.constant.bool() {
// CHECK:           %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT:      return
func.func @torch.constant.bool() {
  %true = torch.constant.bool true
  return
}

// CHECK-LABEL:   func.func @torch.constant.int() {
// CHECK:           %[[C42:.*]] = arith.constant 42 : i64
// CHECK-NEXT:      return
func.func @torch.constant.int() {
  %int = torch.constant.int 42
  return
}

// CHECK-LABEL:   func.func @torch.constant.float() {
// CHECK:           %[[C3_14:.*]] = arith.constant 3.140000e+00 : f64
// CHECK-NEXT:      return
func.func @torch.constant.float() {
  %float = torch.constant.float 3.14
  return
}

// CHECK-LABEL:   func.func @torch.constant.none() {
// CHECK:           %[[NONE:.*]] = torch.constant.none
// CHECK-NEXT:      return
func.func @torch.constant.none() {
  %none = torch.constant.none
  return
}
