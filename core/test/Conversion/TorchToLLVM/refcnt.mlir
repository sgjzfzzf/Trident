//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torch-to-llvm | FileCheck %s
// RUN: trident-core-opt %s --convert-torch-to-llvm -trident-refcnt-debug 2>&1 | FileCheck %s --check-prefix=DBG
//
// These tests exercise the reference-counting logic embedded in
// ConvertTorchToLLVM (see TorchToLLVM.cc::insertRefCounting).
//
// Counting model:
//   * every escape — a converted object used by a terminator operand, i.e.
//     crossing a scope boundary — gets a TVMFFIObjectIncRef;
//   * every object produced by an op directly in the block gets a
//     TVMFFIObjectDecRef (the last use inside the scope consumes the
//     reference).
//
// Inc/Dec of the same object emitted back-to-back at the same insertion point
// cancel out at runtime (net-zero), so the caller always receives a
// reference-count-1 object and no separate pair-elimination pass is needed.
//
// The -trident-refcnt-debug flag (registered by the pass itself, since the
// installed LLVM is a Release build where the generic -debug/-debug-only
// options are compiled out) sets llvm::DebugFlag, which enables the
// LLVM_DEBUG output of the counting decisions for debugging.

// CHECK-LABEL:   func.func @torch.aten.t_escape(
// The aten.t result (a TVMFFIAny) is produced by the FFI call ...
// CHECK:           llvm.call @TVMFFIFunctionCall
// ... its handle is released (function-handle cleanup) ...
// CHECK:           llvm.call @TVMFFIObjectDecRef
// ... and since it escapes through the return, the ref-counting pass emits an
// IncRef (escape) followed by a DecRef (last use in this scope), which cancel
// out at runtime and hand the caller a reference-count-1 tensor.
// CHECK:           llvm.call @TVMFFIObjectIncRef
// CHECK:           llvm.call @TVMFFIObjectDecRef
// CHECK:           return {{.*}} : !torch.vtensor<[3,2],f32>
func.func @torch.aten.t_escape(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL:   func.func @torch.aten.t_chain(
// %0 is produced and consumed inside this scope: it gets only a DecRef (no
// IncRef).  %1 escapes through the return: IncRef + DecRef back-to-back.
// Order: function-handle DecRef, then IncRef(%1 escape), then one DecRef per
// in-scope object (%0, %1).
// CHECK:           llvm.call @TVMFFIObjectDecRef
// CHECK:           llvm.call @TVMFFIObjectIncRef
// CHECK-COUNT-2:   llvm.call @TVMFFIObjectDecRef
// CHECK:           return {{.*}} : !torch.vtensor<[2,3],f32>
func.func @torch.aten.t_chain(%arg0: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %0 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  %1 = torch.aten.t %0 : !torch.vtensor<[3,2],f32> -> !torch.vtensor<[2,3],f32>
  return %1 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL:   func.func @torch.prim.if_aliasing(
// Both branches produce a fresh object that escapes through the yield: each
// branch gets a symmetric IncRef + DecRef pair, and the return of the if
// result (a Torch-typed value, not a converted object) gets no extra counting.
// CHECK:           llvm.call @TVMFFIObjectIncRef
// CHECK:           llvm.call @TVMFFIObjectDecRef
// CHECK:           torch.prim.If.yield {{.*}} : !torch.vtensor<[3,2],f32>
// CHECK:           llvm.call @TVMFFIObjectIncRef
// CHECK:           llvm.call @TVMFFIObjectDecRef
// CHECK:           torch.prim.If.yield {{.*}} : !torch.vtensor<[3,2],f32>
// CHECK:           return {{.*}} : !torch.vtensor<[3,2],f32>
func.func @torch.prim.if_aliasing(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.bool) -> !torch.vtensor<[3,2],f32> {
  %0 = torch.prim.If %arg1 -> (!torch.vtensor<[3,2],f32>) {
    %1 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    torch.prim.If.yield %1 : !torch.vtensor<[3,2],f32>
  } else {
    %2 = torch.aten.t %arg0 : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    torch.prim.If.yield %2 : !torch.vtensor<[3,2],f32>
  }
  return %0 : !torch.vtensor<[3,2],f32>
}

// CHECK-LABEL:   func.func @pass_through_escape(
// A ref-counted func block argument returned directly (a pre-allocated output
// tensor, e.g. flag_gems' mm) is invisible to the conversion table (it is not
// produced by a converted op), but it still escapes the scope and the FFI
// runtime releases the returned handle.  The pass IncRefs it via an
// unrealized conversion cast that the later backend type conversion turns
// into an identity and ReconcileUnrealizedCasts folds away.
// CHECK:           llvm.call @TVMFFIObjectIncRef
// CHECK:           return {{.*}} : !torch.vtensor<[3,2],f32>
func.func @pass_through_escape(%arg0: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[3,2],f32> {
  return %arg0 : !torch.vtensor<[3,2],f32>
}

// The -trident-refcnt-debug decision log: three func-scope escapes (two in
// torch.aten.t_escape/t_chain), then one escape per if-branch.
// DBG: [trident-refcnt] tracking 5 converted object(s)
// DBG: [trident-refcnt] IncRef {{.*}} (escapes through terminator of func.func)
// DBG: [trident-refcnt] DecRef {{.*}} (produced by llvm.load, last use inside scope)
// DBG: [trident-refcnt] IncRef {{.*}} (escapes through terminator of func.func)
// DBG: [trident-refcnt] DecRef {{.*}} (produced by llvm.load, last use inside scope)
// DBG: [trident-refcnt] IncRef {{.*}} (escapes through terminator of torch.prim.If)
// DBG: [trident-refcnt] DecRef {{.*}} (produced by llvm.load, last use inside scope)
// DBG: [trident-refcnt] IncRef {{.*}} (escapes through terminator of torch.prim.If)
// DBG: [trident-refcnt] DecRef {{.*}} (produced by llvm.load, last use inside scope)
// DBG: [trident-refcnt] IncRef {{.*}} (block arg escapes through terminator of func.func)
