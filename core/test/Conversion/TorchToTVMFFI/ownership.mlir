//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -trident-convert-torch-to-scf -convert-torch-to-tvm-ffi -apply-object-ownership | FileCheck %s

// Each branch of torch.prim.If is processed with its own Region-local Set.
// The tensor produced by torch.aten.t is IncRef'd for the branch yield and
// DecRef'd for the branch-local ownership.
// CHECK-LABEL: func.func @nested_if
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[COND:[a-zA-Z0-9_]+]]: !tvm_ffi.bool) -> !tvm_ffi.tensor {
// CHECK-NOT: torch.prim.If
// CHECK-NOT: !torch.vtensor
// CHECK: %[[NATIVE_COND:[a-zA-Z0-9_]+]] = tvm_ffi.get %[[COND]] : !tvm_ffi.bool -> i1
// CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[NATIVE_COND]] -> (!tvm_ffi.tensor) {
// CHECK: %[[TRUE_FUNCTION:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.t" : !tvm_ffi.function
// CHECK: %[[TRUE_TRANSPOSE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[TRUE_FUNCTION]](%[[INPUT]]) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[TRUE_TRANSPOSE]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[TRUE_TRANSPOSE]] : !tvm_ffi.tensor
// CHECK-NEXT: scf.yield %[[TRUE_TRANSPOSE]] : !tvm_ffi.tensor
// CHECK: %[[FALSE_FUNCTION:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.t" : !tvm_ffi.function
// CHECK: %[[FALSE_TRANSPOSE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FALSE_FUNCTION]](%[[INPUT]]) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FALSE_TRANSPOSE]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FALSE_TRANSPOSE]] : !tvm_ffi.tensor
// CHECK-NEXT: scf.yield %[[FALSE_TRANSPOSE]] : !tvm_ffi.tensor
// CHECK: return %[[RESULT]] : !tvm_ffi.tensor
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
// CHECK: %[[FFI_FUNCTION:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.t" : !tvm_ffi.function
// CHECK-NEXT: %[[FFI_RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FFI_FUNCTION]](%arg0) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[FFI_RESULT]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[FFI_RESULT]] : !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.return %[[FFI_RESULT]] : !tvm_ffi.tensor
tvm_ffi.func @ffi_return(%arg0: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0
      : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  tvm_ffi.return %0 : !torch.vtensor<[3,2],f32>
}

// An scf.if that forwards owned call results into an ABI container releases
// the static tensor ownership; the dynamic container is outside this static
// Object ownership analysis.
// CHECK-LABEL: func.func @control_flow_return
// CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1) -> !tvm_ffi.any {
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = scf.if %[[COND]] -> (!tvm_ffi.any) {
// CHECK: %[[THEN_CALL:[a-zA-Z0-9_]+]] = func.call @make_tensor() : () -> !tvm_ffi.tensor
// CHECK-NEXT: %[[THEN_CAST:[a-zA-Z0-9_]+]] = tvm_ffi.cast %[[THEN_CALL]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[THEN_CAST]] : !tvm_ffi.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[THEN_CALL]] : !tvm_ffi.tensor
// CHECK-NEXT: scf.yield %[[THEN_CAST]] : !tvm_ffi.any
// CHECK: %[[ELSE_CALL:[a-zA-Z0-9_]+]] = func.call @make_tensor() : () -> !tvm_ffi.tensor
// CHECK-NEXT: %[[ELSE_CAST:[a-zA-Z0-9_]+]] = tvm_ffi.cast %[[ELSE_CALL]]
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ELSE_CAST]] : !tvm_ffi.any
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELSE_CALL]] : !tvm_ffi.tensor
// CHECK-NEXT: scf.yield %[[ELSE_CAST]] : !tvm_ffi.any
// CHECK: return %[[RESULT]] : !tvm_ffi.any
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
// CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK: %[[BORROWED_RESULT:[a-zA-Z0-9_]+]] = scf.if %[[COND]] -> (!tvm_ffi.tensor) {
// CHECK: scf.yield %[[INPUT]] : !tvm_ffi.tensor
// CHECK: scf.yield %[[INPUT]] : !tvm_ffi.tensor
// CHECK: return %[[BORROWED_RESULT]] : !tvm_ffi.tensor
func.func @borrowed_control_flow(%cond: i1, %value: !tvm_ffi.tensor)
    -> !tvm_ffi.tensor {
  %0 = scf.if %cond -> (!tvm_ffi.tensor) {
    scf.yield %value : !tvm_ffi.tensor
  } else {
    scf.yield %value : !tvm_ffi.tensor
  }
  return %0 : !tvm_ffi.tensor
}

// Existing explicit reference operations participate in the same ledger. A
// balanced owned result must not receive another terminator cleanup.
// CHECK-LABEL: func.func @explicit_balance
// CHECK: %[[BALANCED:[a-zA-Z0-9_]+]] = call @make_tensor() : () -> !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[BALANCED]]
// CHECK-NEXT: return
func.func @explicit_balance() {
  %value = func.call @make_tensor() : () -> !tvm_ffi.tensor
  tvm_ffi.ObjectDecRef %value : !tvm_ffi.tensor
  return
}

// An explicit increment adds another credit to the same ownership ledger, so
// both the original and retained references are released at the terminator.
// CHECK-LABEL: func.func @explicit_increment
// CHECK: %[[RETAINED:[a-zA-Z0-9_]+]] = call @make_tensor() : () -> !tvm_ffi.tensor
// CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RETAINED]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RETAINED]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RETAINED]]
// CHECK-NEXT: return
func.func @explicit_increment() {
  %value = func.call @make_tensor() : () -> !tvm_ffi.tensor
  tvm_ffi.ObjectIncRef %value : !tvm_ffi.tensor
  return
}

// FunctionCall consumes the owned global-function handle. The ownership
// analysis must not schedule a second semantic DecRef for it.
// CHECK-LABEL: func.func @consumed_function_handle
// CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "test.make_tensor" : !tvm_ffi.function
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]() : () -> !tvm_ffi.tensor
// CHECK-NOT: tvm_ffi.ObjectDecRef %[[HANDLE]]
// CHECK: tvm_ffi.ObjectIncRef %[[VALUE]]
// CHECK-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]]
// CHECK-NEXT: return %[[VALUE]]
func.func @consumed_function_handle() -> !tvm_ffi.tensor {
  %function = tvm_ffi.FunctionGetGlobal "test.make_tensor"
      : !tvm_ffi.function
  %value = tvm_ffi.FunctionCall %function()
      : () -> !tvm_ffi.tensor
  return %value : !tvm_ffi.tensor
}
