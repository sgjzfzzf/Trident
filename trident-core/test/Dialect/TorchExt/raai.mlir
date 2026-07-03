// RUN: trident-core-opt %s --raai -split-input-file | FileCheck %s
//
// raai — Release Allocated Intermediate variables.
// Inserts torchext.ObjectIncRef / ObjectDecRef before the terminator of
// single-block func::FuncOp regions.

// UNALLOC: single allocation, not yielded → DecRef only.
// CHECK-LABEL: @list_construct_no_return
func.func @list_construct_no_return(%x: !torch.int, %y: !torch.int) {
  %list = torch.prim.ListConstruct %x, %y : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:      torchext.ObjectDecRef %[[LIST:.*]] : !torch.list<int>
  // CHECK-NOT:  torchext.ObjectIncRef
  return
}

// -----

// ALLOC + YIELD: aten op tensor yielded → IncRef + DecRef.
// CHECK-LABEL: @aten_tensor_returned
func.func @aten_tensor_returned(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  // CHECK:      torchext.ObjectIncRef %[[RES:.*]] : !torch.vtensor<[?,?],f64>
  // CHECK-NEXT: torchext.ObjectDecRef %[[RES]] : !torch.vtensor<[?,?],f64>
  return %0 : !torch.vtensor<[?,?],f64>
}

// -----

// LITERAL + YIELD: vtensor literal is also a newly created tensor.
// CHECK-LABEL: @vtensor_literal_returned
func.func @vtensor_literal_returned() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<1.250000e+00> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  // CHECK:      torchext.ObjectIncRef %[[LIT:.*]] : !torch.vtensor<[2,3],f32>
  // CHECK-NEXT: torchext.ObjectDecRef %[[LIT]] : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----

// MIXED: list (not yielded) + tensor (yielded).
//   IncRef only for the yielded tensor; DecRef for both.
// CHECK-LABEL: @list_and_tensor_mixed
func.func @list_and_tensor_mixed(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %c1 = torch.constant.int 1
  %c2 = torch.constant.int 2
  %list = torch.prim.ListConstruct %c1, %c2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  // CHECK:      torchext.ObjectIncRef %[[TENSOR:.*]] : !torch.vtensor<[?,?],f64>
  // CHECK-NEXT: torchext.ObjectDecRef %[[LIST:.*]] : !torch.list<int>
  // CHECK-NEXT: torchext.ObjectDecRef %[[TENSOR]] : !torch.vtensor<[?,?],f64>
  return %0 : !torch.vtensor<[?,?],f64>
}

// -----

// EXTERNAL: arg passthrough → only IncRef, no internal allocation to release.
// CHECK-LABEL: @arg_passthrough
func.func @arg_passthrough(%t: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  // CHECK:      torchext.ObjectIncRef %[[T:.*]] : !torch.vtensor<[200,200,26],f64>
  // CHECK-NOT:  torchext.ObjectDecRef
  return %t : !torch.vtensor<[200,200,26],f64>
}

// -----

// TWO ATEN, ONE YIELD: %0 not yielded, %1 yielded.
//   Alloc order is %0, %1 → DecRef %0, IncRef %1, DecRef %1.
// CHECK-LABEL: @multi_aten_partial_yield
func.func @multi_aten_partial_yield(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  %1 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  // CHECK:      torchext.ObjectIncRef %[[RET:.*]] : !torch.vtensor<[?,?],f64>
  // CHECK-NEXT: torchext.ObjectDecRef %[[TMP:.*]] : !torch.vtensor<[?,?],f64>
  // CHECK-NEXT: torchext.ObjectDecRef %[[RET]] : !torch.vtensor<[?,?],f64>
  return %1 : !torch.vtensor<[?,?],f64>
}

// -----

// MULTI-BLOCK: pass skips multi-block regions.
// CHECK-LABEL: @multi_block
func.func @multi_block(%x: !torch.int, %y: !torch.int) {
  %list = torch.prim.ListConstruct %x, %y : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK-NOT: torchext.ObjectDecRef
  // CHECK-NOT: torchext.ObjectIncRef
  cf.br ^bb1
^bb1:
  cf.br ^bb2
^bb2:
  return
}

// -----

// NO-OP: no allocations, no yielded values.
// CHECK-LABEL: @empty_body
func.func @empty_body() {
  %c1 = torch.constant.int 1
  // CHECK-NOT: torchext.ObjectDecRef
  // CHECK-NOT: torchext.ObjectIncRef
  return
}

// -----

// MULTI-YIELD: both values allocated internally, both yielded.
// CHECK-LABEL: @multi_yield
func.func @multi_yield(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> (!torch.vtensor<[?,?],f64>, !torch.vtensor<[?,?],f64>) {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  %1 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  // CHECK:      torchext.ObjectIncRef %[[T0:.*]] : !torch.vtensor<[?,?],f64>
  // CHECK-NEXT: torchext.ObjectIncRef %[[T1:.*]] : !torch.vtensor<[?,?],f64>
  // CHECK-NEXT: torchext.ObjectDecRef %[[T0]] : !torch.vtensor<[?,?],f64>
  // CHECK-NEXT: torchext.ObjectDecRef %[[T1]] : !torch.vtensor<[?,?],f64>
  return %0, %1 : !torch.vtensor<[?,?],f64>, !torch.vtensor<[?,?],f64>
}

// -----

// EXTERNAL + INTERNAL both yielded.
//   Only the internal allocation gets a DecRef.
// CHECK-LABEL: @arg_and_alloc_yielded
func.func @arg_and_alloc_yielded(%t: !torch.vtensor<[200,200,26],f64>, %shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> (!torch.vtensor<[200,200,26],f64>, !torch.vtensor<[?,?],f64>) {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  // CHECK:      torchext.ObjectIncRef %[[EXT:.*]] : !torch.vtensor<[200,200,26],f64>
  // CHECK-NEXT: torchext.ObjectIncRef %[[INT:.*]] : !torch.vtensor<[?,?],f64>
  // CHECK-NEXT: torchext.ObjectDecRef %[[INT]] : !torch.vtensor<[?,?],f64>
  return %t, %0 : !torch.vtensor<[200,200,26],f64>, !torch.vtensor<[?,?],f64>
}
