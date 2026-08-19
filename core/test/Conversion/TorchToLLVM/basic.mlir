//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -decompose-tvm-ffi -convert-tvm-ffi-to-llvm | FileCheck %s

// === Section 1: no-result / single-result returns ===

// void_func: no results -> nothing is stored into %arg3; the function just
// returns 0.
// CHECK-LABEL: llvm.func @__tvm_ffi_void_func(
// CHECK: llvm.br ^bb1
// CHECK: llvm.return %[[STATUS:.*]] : i32
tvm_ffi.func @void_func() {
  tvm_ffi.return
}

// -----

// make_int: a single scalar result is cast to its converted (TVMFFIAny)
// type and stored directly into %arg3.
// [[ANY]] is defined here (first use of the TVMFFIAny struct type) and
// referenced by every test below.
// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK: builtin.unrealized_conversion_cast %int42 : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[RESULT:.*]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @make_int() -> !torch.int {
  %0 = torch.constant.int 42
  tvm_ffi.return %0 : !torch.int
}

// -----

// === Section 2: argument marshalling ===

// print_int: an !torch.int argument is unpacked from the arg array — GEP into
// the slot, load the TVMFFIAny, cast to the Torch type.
// CHECK-LABEL: llvm.func @__tvm_ffi_print_int(
// CHECK: llvm.getelementptr %arg1[0]
// CHECK: %[[ANY:.*]] = llvm.load %[[ARG_SLOT:.*]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[INT:.*]] = builtin.unrealized_conversion_cast %[[ANY]] : !llvm.struct<(i32, i32, i64)> to !torch.int
tvm_ffi.func @print_int(%arg0: !torch.int) {
  tvm_ffi.return
}

// -----

// identity_bool: an !torch.bool argument is unpacked and returned unchanged.
// On return the Torch value is cast back to TVMFFIAny and stored into %arg3.
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_bool(
// CHECK: builtin.unrealized_conversion_cast %[[BOOL_ANY:.*]] to !torch.bool
// CHECK: builtin.unrealized_conversion_cast %[[BOOL:.*]] : !torch.bool to !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[BOOL_RESULT:.*]], %arg3
tvm_ffi.func @identity_bool(%arg0: !torch.bool) -> !torch.bool {
  tvm_ffi.return %arg0 : !torch.bool
}

// -----

// identity_float: same round-trip as identity_bool, for !torch.float.
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_float(
// CHECK: %[[FLOAT_ANY:.*]] = llvm.load %[[FLOAT_SLOT:.*]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[FLOAT_RESULT:.*]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @identity_float(%arg0: !torch.float) -> !torch.float {
  tvm_ffi.return %arg0 : !torch.float
}

// -----

// tensor_func: an !torch.tensor argument is unpacked; nothing is returned.
tvm_ffi.func @tensor_func(%arg0: !torch.tensor) {
  tvm_ffi.return
}

// -----

// make_tensor: a tensor produced by torch.aten.empty.memory_format is
// returned; the single TVMFFIAny result is stored into %arg3.
// CHECK-LABEL: llvm.func @__tvm_ffi_make_tensor(
// CHECK: torch.aten.empty.memory_format
// CHECK: llvm.store %[[TENSOR_RESULT:.*]], %arg3
tvm_ffi.func @make_tensor() -> !torch.tensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %shape = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %tensor = torch.aten.empty.memory_format %shape, %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor
  tvm_ffi.return %tensor : !torch.tensor
}

// -----

// === Section 3: multi-value (tuple) returns ===
//
// The FFI C ABI carries a single result slot, so multi-result functions pack
// every value into an ffi.Array container (kTVMFFIArray == 71) and store it
// into %arg3; each object-typed element is DecRef'd after packing.

// multi_return_tensor_bool: (tensor, bool) — the flash-attention
// seqlenq_ngroups_swapped scenario.  Exactly one element DecRef is emitted
// (the tensor); the bool is a scalar and must not be ref-counted.
// * Cast each result to its converted TVMFFIAny type.
// * Allocate the per-result slot array and fill it.
// * Look up the ffi.Array handle and call it with the slot array.
// * Re-tag the container handle as a kTVMFFIArray (71) TVMFFIAny.
// * DecRef the tensor element (the container owns its reference); the bool
//     is a scalar and must not appear here.
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_tensor_bool(
// CHECK: llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[ELEMENT_SLOTS:.*]] = llvm.alloca %[[ELEMENT_COUNT:.*]] x !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.call @TVMFFIFunctionGetGlobal
// CHECK: llvm.call @TVMFFIFunctionCall
// CHECK: llvm.mlir.constant(71 : i32) : i32
// CHECK: llvm.store %[[ARRAY_RESULT:.*]], %arg3
// CHECK: llvm.call @TVMFFIObjectDecRef
tvm_ffi.func @multi_return_tensor_bool(%arg0: !torch.tensor, %arg1: !torch.bool) -> (!torch.tensor, !torch.bool) {
  tvm_ffi.return %arg0, %arg1 : !torch.tensor, !torch.bool
}

// -----

// multi_return_int_int: scalar-only tuple.  No element DecRef may be emitted
// between the container store and the return — only the ffi.Array function
// handle DecRef inside the call helper.
tvm_ffi.func @multi_return_int_int(%arg0: !torch.int, %arg1: !torch.int) -> (!torch.int, !torch.int) {
  tvm_ffi.return %arg0, %arg1 : !torch.int, !torch.int
}

// -----

// multi_return_three_tensors: three object elements → three element DecRefs
// after the container store.
// * Allocate three slots and fill each with a casted tensor.
// * ffi.Array call helper + kTVMFFIArray re-tag + store into %arg3.
// The input tensors are borrowed from the caller, so only the temporary
// ffi.Array function handle is released here; no element DecRef is emitted.
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_three_tensors(
// CHECK: llvm.mlir.constant(3 : i64) : i64
// CHECK: llvm.mlir.constant(71 : i32) : i32
// CHECK: llvm.store %[[THREE_ARRAY_RESULT:.*]], %arg3
tvm_ffi.func @multi_return_three_tensors(%arg0: !torch.tensor, %arg1: !torch.tensor, %arg2: !torch.tensor) -> (!torch.tensor, !torch.tensor, !torch.tensor) {
  tvm_ffi.return %arg0, %arg1, %arg2 : !torch.tensor, !torch.tensor, !torch.tensor
}

// -----

// A semantic call to a local TVMFFI function is lowered to an ABI-level
// func.call. The later FuncToLLVM conversion turns this into llvm.call.
// CHECK-LABEL: llvm.func @__tvm_ffi_callee(
// CHECK-LABEL: llvm.func @__tvm_ffi_caller(
// CHECK: llvm.call @__tvm_ffi_callee
tvm_ffi.func @callee(%arg0: !tvm_ffi.int) -> !tvm_ffi.int {
  tvm_ffi.return %arg0 : !tvm_ffi.int
}

tvm_ffi.func @caller(%arg0: !tvm_ffi.int) -> !tvm_ffi.int {
  %0 = tvm_ffi.call @callee(%arg0) : (!tvm_ffi.int) -> !tvm_ffi.int
  tvm_ffi.return %0 : !tvm_ffi.int
}