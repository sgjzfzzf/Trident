//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --raai -split-input-file | FileCheck %s
//
// raai — Release Allocated Intermediate variables.
// Inserts torchext.ObjectIncRef / ObjectDecRef before the terminator of
// single-block func::FuncOp regions.

// -----

// test_01: ListConstruct not yielded -> DecRef only.
// CHECK-LABEL: @test_01
// CHECK:       torchext.ObjectDecRef %[[LIST:.*]] : !torch.list<int>
// CHECK-NOT:   torchext.ObjectIncRef
func.func @test_01(%x: !torch.int, %y: !torch.int) {
  %list = torch.prim.ListConstruct %x, %y : (!torch.int, !torch.int) -> !torch.list<int>
  return
}

// -----

// test_02: aten op tensor yielded -> IncRef + DecRef.
// CHECK-LABEL: @test_02
// CHECK:       torchext.ObjectIncRef %[[T:.*]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectDecRef %[[T]] : !torch.vtensor<[?,?],f64>
func.func @test_02(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  return %0 : !torch.vtensor<[?,?],f64>
}

// -----

// test_03: vtensor literal yielded -> IncRef + DecRef.
// CHECK-LABEL: @test_03
// CHECK:       torchext.ObjectIncRef %[[LIT:.*]] : !torch.vtensor<[2,3],f32>
// CHECK-NEXT:  torchext.ObjectDecRef %[[LIT]] : !torch.vtensor<[2,3],f32>
func.func @test_03() -> !torch.vtensor<[2,3],f32> {
  %0 = torch.vtensor.literal(dense<1.250000e+00> : tensor<2x3xf32>) : !torch.vtensor<[2,3],f32>
  return %0 : !torch.vtensor<[2,3],f32>
}

// -----

// test_04: list (not yielded) + tensor (yielded) -> IncRef tensor, DecRef both.
// CHECK-LABEL: @test_04
// CHECK:       torchext.ObjectIncRef %[[TENSOR:.*]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectDecRef %[[LIST:.*]] : !torch.list<int>
// CHECK-NEXT:  torchext.ObjectDecRef %[[TENSOR]] : !torch.vtensor<[?,?],f64>
func.func @test_04(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %c1 = torch.constant.int 1
  %c2 = torch.constant.int 2
  %list = torch.prim.ListConstruct %c1, %c2 : (!torch.int, !torch.int) -> !torch.list<int>
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  return %0 : !torch.vtensor<[?,?],f64>
}

// -----

// test_05: arg passthrough -> IncRef only, no internal allocation.
// CHECK-LABEL: @test_05
// CHECK:       torchext.ObjectIncRef %[[ARG:.*]] : !torch.vtensor<[200,200,26],f64>
// CHECK-NOT:   torchext.ObjectDecRef
func.func @test_05(%t: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  return %t : !torch.vtensor<[200,200,26],f64>
}

// -----

// test_06: two aten allocs, only %1 yielded -> IncRef %1, DecRef both.
// CHECK-LABEL: @test_06
// CHECK:       torchext.ObjectIncRef %[[YLD:.*]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectDecRef %[[TMP:.*]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectDecRef %[[YLD]] : !torch.vtensor<[?,?],f64>
func.func @test_06(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> !torch.vtensor<[?,?],f64> {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  %1 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  return %1 : !torch.vtensor<[?,?],f64>
}

// -----

// test_07: multi-block region -> pass skipped, no IncRef/DecRef.
// CHECK-LABEL: @test_07
// CHECK-NOT:   torchext.ObjectDecRef
// CHECK-NOT:   torchext.ObjectIncRef
func.func @test_07(%x: !torch.int, %y: !torch.int) {
  %list = torch.prim.ListConstruct %x, %y : (!torch.int, !torch.int) -> !torch.list<int>
  cf.br ^bb1
^bb1:
  cf.br ^bb2
^bb2:
  return
}

// -----

// test_08: no allocations, no yielded values -> no IncRef/DecRef.
// CHECK-LABEL: @test_08
// CHECK-NOT:   torchext.ObjectDecRef
// CHECK-NOT:   torchext.ObjectIncRef
func.func @test_08() {
  %c1 = torch.constant.int 1
  return
}

// -----

// test_09: both values internally allocated and yielded -> IncRef+DecRef each.
// CHECK-LABEL: @test_09
// CHECK:       torchext.ObjectIncRef %[[T0:.*]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectIncRef %[[T1:.*]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectDecRef %[[T0]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectDecRef %[[T1]] : !torch.vtensor<[?,?],f64>
func.func @test_09(%shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> (!torch.vtensor<[?,?],f64>, !torch.vtensor<[?,?],f64>) {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  %1 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  return %0, %1 : !torch.vtensor<[?,?],f64>, !torch.vtensor<[?,?],f64>
}

// -----

// test_10: arg (no DecRef) + internal alloc (DecRef) both yielded.
// CHECK-LABEL: @test_10
// CHECK:       torchext.ObjectIncRef %[[EXT:.*]] : !torch.vtensor<[200,200,26],f64>
// CHECK-NEXT:  torchext.ObjectIncRef %[[INT:.*]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectDecRef %[[INT]] : !torch.vtensor<[?,?],f64>
func.func @test_10(%t: !torch.vtensor<[200,200,26],f64>, %shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> (!torch.vtensor<[200,200,26],f64>, !torch.vtensor<[?,?],f64>) {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  return %t, %0 : !torch.vtensor<[200,200,26],f64>, !torch.vtensor<[?,?],f64>
}

// -----

// test_11: TupleConstruct not yielded -> DecRef only.
// CHECK-LABEL: @test_11
// CHECK:       torchext.ObjectDecRef %[[TUP:.*]] : !torch.tuple<int, int>
// CHECK-NOT:   torchext.ObjectIncRef
func.func @test_11(%x: !torch.int, %y: !torch.int) {
  %tup = torch.prim.TupleConstruct %x, %y : !torch.int, !torch.int -> !torch.tuple<int, int>
  return
}

// -----

// test_12: TupleConstruct yielded -> IncRef + DecRef.
// CHECK-LABEL: @test_12
// CHECK:       torchext.ObjectIncRef %[[TUP:.*]] : !torch.tuple<int, int>
// CHECK-NEXT:  torchext.ObjectDecRef %[[TUP]] : !torch.tuple<int, int>
func.func @test_12(%x: !torch.int, %y: !torch.int) -> !torch.tuple<int, int> {
  %tup = torch.prim.TupleConstruct %x, %y : !torch.int, !torch.int -> !torch.tuple<int, int>
  return %tup : !torch.tuple<int, int>
}

// -----

// test_13: CastOp (torch.derefine) is skipped — no DecRef/IncRef inserted.
// CHECK-LABEL: @test_13
// CHECK:       %[[OPT:.*]] = torch.derefine %[[T:.*]] : !torch.vtensor<[?,?],f64> to !torch.optional<vtensor<[?,?],f64>>
// CHECK-NOT:   torchext.Object
func.func @test_13(%t: !torch.vtensor<[?,?],f64>) {
  %opt = torch.derefine %t : !torch.vtensor<[?,?],f64> to !torch.optional<vtensor<[?,?],f64>>
  return
}

// -----

// test_14: CastOp (torch.derefine) skipped — only IncRef from terminator, no DecRef.
// CHECK-LABEL: @test_14
// CHECK:       torchext.ObjectIncRef %[[OPT:.*]] : !torch.optional<vtensor<[?,?],f64>>
// CHECK-NOT:   torchext.ObjectDecRef
func.func @test_14(%t: !torch.vtensor<[?,?],f64>) -> !torch.optional<vtensor<[?,?],f64>> {
  %opt = torch.derefine %t : !torch.vtensor<[?,?],f64> to !torch.optional<vtensor<[?,?],f64>>
  return %opt : !torch.optional<vtensor<[?,?],f64>>
}

// -----

// test_15: CastOp (torch.derefine from none) skipped — no DecRef/IncRef.
// CHECK-LABEL: @test_15
// CHECK:       %[[OPT:.*]] = torch.derefine %[[NONE:.*]] : !torch.none to !torch.optional<vtensor<[?,?],f64>>
// CHECK-NOT:   torchext.Object
func.func @test_15() {
  %none = torch.constant.none
  %opt = torch.derefine %none : !torch.none to !torch.optional<vtensor<[?,?],f64>>
  return
}

// -----

// test_16: tuple + list + tensor + optional mixed in one function.
//   CastOp (torch.derefine) skipped; optional gets IncRef but no DecRef.
// CHECK-LABEL: @test_16
// CHECK:       torchext.ObjectIncRef %[[TUP:.*]] : !torch.tuple<int, int>
// CHECK-NEXT:  torchext.ObjectIncRef %[[T0:.*]] : !torch.vtensor<[?,?],f64>
// CHECK-NEXT:  torchext.ObjectIncRef %[[OPT:.*]] : !torch.optional<vtensor<[?,?],f64>>
// CHECK-NEXT:  torchext.ObjectDecRef %[[TUP]] : !torch.tuple<int, int>
// CHECK-NEXT:  torchext.ObjectDecRef %[[LIST:.*]] : !torch.list<int>
// CHECK-NEXT:  torchext.ObjectDecRef %[[T0]] : !torch.vtensor<[?,?],f64>
// CHECK-NOT:   torchext.ObjectDecRef %[[OPT]]
func.func @test_16(%x: !torch.int, %y: !torch.int, %t: !torch.vtensor<[?,?],f64>, %shape: !torch.list<int>, %dtype: !torch.int, %device: !torch.Device) -> (!torch.tuple<int, int>, !torch.vtensor<[?,?],f64>, !torch.optional<vtensor<[?,?],f64>>) {
  %none = torch.constant.none
  %layout = torch.constant.int 0
  %tup = torch.prim.TupleConstruct %x, %y : !torch.int, !torch.int -> !torch.tuple<int, int>
  %list = torch.prim.ListConstruct %x, %y : (!torch.int, !torch.int) -> !torch.list<int>
  %t0 = torch.aten.empty.memory_format %shape, %dtype, %layout, %device, %none, %none : !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.none, !torch.none -> !torch.vtensor<[?,?],f64>
  %opt = torch.derefine %t : !torch.vtensor<[?,?],f64> to !torch.optional<vtensor<[?,?],f64>>
  return %tup, %t0, %opt : !torch.tuple<int, int>, !torch.vtensor<[?,?],f64>, !torch.optional<vtensor<[?,?],f64>>
}
