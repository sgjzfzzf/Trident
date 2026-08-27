//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// Each branch of torch.prim.If is processed with its own Region-local Set.
// The tensor produced by torch.aten.t is IncRef'd for the branch yield and
// DecRef'd for the branch-local ownership.
// CHECK-LABEL: func.func @nested_if
// CHECK: torch.prim.If
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: torch.prim.If.yield
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: torch.prim.If.yield
func.func @nested_if(%arg0: !torch.vtensor<[2,3],f32>, %cond: !torch.bool)
    -> !torch.vtensor<[3,2],f32> {
  %0 = torch.prim.If %cond -> (!torch.vtensor<[3,2],f32>) {
    %1 = torch.aten.t %arg0
        : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    torch.prim.If.yield %1 : !torch.vtensor<[3,2],f32>
  } else {
    %2 = torch.aten.t %arg0
        : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    torch.prim.If.yield %2 : !torch.vtensor<[3,2],f32>
  }
  return %0 : !torch.vtensor<[3,2],f32>
}

// The TVMFFI function body is also a Region root, so its return is handled
// by the same terminator pattern.
// CHECK-LABEL: tvm_ffi.func @ffi_return
// CHECK-SAME: %arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: tvm_ffi.return
tvm_ffi.func @ffi_return(%arg0: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0
      : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  tvm_ffi.return %0 : !torch.vtensor<[3,2],f32>
}

// An scf.if that forwards owned call results must not add a net reference at
// the outer return. The balancing DecRef prevents the returned tensor from
// leaking while preserving a live reference for the caller.
// CHECK-LABEL: func.func @control_flow_return
// CHECK: %[[RESULT:.*]] = scf.if
// CHECK: %[[THEN_CALL:.*]] = func.call @make_tensor
// CHECK-NEXT: %[[THEN_CAST:.*]] = tvm_ffi.cast %[[THEN_CALL]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[THEN_CALL]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[THEN_CALL]] : !tvm_ffi.tensor
// CHECK-NEXT: scf.yield %[[THEN_CAST]] : !tvm_ffi.any
// CHECK: %[[ELSE_CALL:.*]] = func.call @make_tensor
// CHECK-NEXT: %[[ELSE_CAST:.*]] = tvm_ffi.cast %[[ELSE_CALL]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ELSE_CALL]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELSE_CALL]] : !tvm_ffi.tensor
// CHECK-NEXT: scf.yield %[[ELSE_CAST]] : !tvm_ffi.any
// CHECK: tvm_ffi.ObjectIncRef %[[RESULT]] : !tvm_ffi.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]] : !tvm_ffi.any
// CHECK-NEXT: return %[[RESULT]] : !tvm_ffi.any
func.func private @make_tensor() -> !tvm_ffi.tensor

func.func @control_flow_return(%cond: i1) -> !tvm_ffi.any {
  %0 = scf.if %cond -> (!tvm_ffi.any) {
    %1 = func.call @make_tensor() : () -> !tvm_ffi.tensor
    %2 = tvm_ffi.cast %1 : !tvm_ffi.tensor -> !tvm_ffi.any
    scf.yield %2 : !tvm_ffi.any
  } else {
    %1 = func.call @make_tensor() : () -> !tvm_ffi.tensor
    %2 = tvm_ffi.cast %1 : !tvm_ffi.tensor -> !tvm_ffi.any
    scf.yield %2 : !tvm_ffi.any
  }
  return %0 : !tvm_ffi.any
}

// A borrowed branch input gains one reference at the yield. It must not be
// released as branch-local ownership.
// CHECK-LABEL: func.func @borrowed_control_flow
// CHECK: %[[BORROWED_RESULT:.*]] = scf.if
// CHECK: tvm_ffi.ObjectIncRef %[[BORROWED:.*]] : !tvm_ffi.tensor
// CHECK-NEXT: scf.yield %[[BORROWED]] : !tvm_ffi.tensor
// CHECK: tvm_ffi.ObjectIncRef %[[BORROWED]] : !tvm_ffi.tensor
// CHECK-NEXT: scf.yield %[[BORROWED]] : !tvm_ffi.tensor
// CHECK: tvm_ffi.ObjectIncRef %[[BORROWED_RESULT]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[BORROWED_RESULT]] : !tvm_ffi.tensor
// CHECK-NEXT: return %[[BORROWED_RESULT]] : !tvm_ffi.tensor
func.func @borrowed_control_flow(%cond: i1, %value: !tvm_ffi.tensor)
    -> !tvm_ffi.tensor {
  %0 = scf.if %cond -> (!tvm_ffi.tensor) {
    scf.yield %value : !tvm_ffi.tensor
  } else {
    scf.yield %value : !tvm_ffi.tensor
  }
  return %0 : !tvm_ffi.tensor
}
