//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s

// === Section 1: no-result / single-result returns ===

// void_func: no results -> nothing is stored into %arg3; the function just
// returns 0.
// CHECK-LABEL: llvm.func @__tvm_ffi_void_func(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @void_func() {
  tvm_ffi.return
}

// -----

// make_int: a single scalar result is cast to its converted (TVMFFIAny)
// type and stored directly into %arg3.
// [[ANY]] is defined here (first use of the TVMFFIAny struct type) and
// referenced by every test below.
// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    %int42 = torch.constant.int 42
// CHECK-NEXT:    [[CASTED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast %int42 : !torch.int to [[ANY:!llvm\.struct<\(i32, i32, i64\)>]]
// CHECK-NEXT:    llvm.store [[CASTED]], %arg3 : [[ANY]], !llvm.ptr
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @make_int() -> !torch.int {
  %0 = torch.constant.int 42
  tvm_ffi.return %0 : !torch.int
}

// -----

// === Section 2: argument marshalling ===

// print_int: an !torch.int argument is unpacked from the arg array — GEP into
// the slot, load the TVMFFIAny, cast to the Torch type.
// CHECK-LABEL: llvm.func @__tvm_ffi_print_int(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    [[LOADED:%[a-z0-9]+]] = llvm.load [[ARG_GEP]] : !llvm.ptr -> [[ANY]]
// CHECK-NEXT:    [[CASTED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : [[ANY]] to !torch.int
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @print_int(%arg0: !torch.int) {
  tvm_ffi.return
}

// -----

// identity_bool: an !torch.bool argument is unpacked and returned unchanged.
// On return the Torch value is cast back to TVMFFIAny and stored into %arg3.
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_bool(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    [[LOADED:%[a-z0-9]+]] = llvm.load [[ARG_GEP]] : !llvm.ptr -> [[ANY]]
// CHECK-NEXT:    [[CASTED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : [[ANY]] to !torch.bool
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    [[REPACKED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[CASTED]] : !torch.bool to [[ANY]]
// CHECK-NEXT:    llvm.store [[REPACKED]], %arg3 : [[ANY]], !llvm.ptr
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @identity_bool(%arg0: !torch.bool) -> !torch.bool {
  tvm_ffi.return %arg0 : !torch.bool
}

// -----

// identity_float: same round-trip as identity_bool, for !torch.float.
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_float(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    [[LOADED:%[a-z0-9]+]] = llvm.load [[ARG_GEP]] : !llvm.ptr -> [[ANY]]
// CHECK-NEXT:    [[CASTED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : [[ANY]] to !torch.float
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    [[REPACKED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[CASTED]] : !torch.float to [[ANY]]
// CHECK-NEXT:    llvm.store [[REPACKED]], %arg3 : [[ANY]], !llvm.ptr
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @identity_float(%arg0: !torch.float) -> !torch.float {
  tvm_ffi.return %arg0 : !torch.float
}

// -----

// tensor_func: an !torch.tensor argument is unpacked; nothing is returned.
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_func(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    [[LOADED:%[a-z0-9]+]] = llvm.load [[ARG_GEP]] : !llvm.ptr -> [[ANY]]
// CHECK-NEXT:    [[CASTED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : [[ANY]] to !torch.tensor
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @tensor_func(%arg0: !torch.tensor) {
  tvm_ffi.return
}

// -----

// make_tensor: a tensor produced by torch.aten.empty.memory_format is
// returned; the single TVMFFIAny result is stored into %arg3.
// CHECK-LABEL: llvm.func @__tvm_ffi_make_tensor(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    %int3 = torch.constant.int 3
// CHECK-NEXT:    %int4 = torch.constant.int 4
// CHECK-NEXT:    [[LIST:%[a-z0-9]+]] = torch.prim.ListConstruct %int3, %int4
// CHECK-SAME:      : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:    %none = torch.constant.none
// CHECK-NEXT:    [[TENSOR:%[a-z0-9]+]] = torch.aten.empty.memory_format [[LIST]]
// CHECK-NEXT:    [[CASTED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[TENSOR]] : !torch.tensor to [[ANY]]
// CHECK-NEXT:    llvm.store [[CASTED]], %arg3 : [[ANY]], !llvm.ptr
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
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
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_tensor_bool(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// --- Cast each result to its converted TVMFFIAny type.
// CHECK-NEXT:    [[T_ANY:%[a-z0-9]+]] = builtin.unrealized_conversion_cast {{%.*}} : !torch.tensor to [[ANY]]
// CHECK-NEXT:    [[B_ANY:%[a-z0-9]+]] = builtin.unrealized_conversion_cast {{%.*}} : !torch.bool to [[ANY]]
// --- Allocate the per-result slot array and fill it.
// CHECK-NEXT:    %[[N:[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    [[SLOTS:%[a-z0-9]+]] = llvm.alloca %[[N]] x [[ANY]] : (i64) -> !llvm.ptr
// CHECK-NEXT:    [[SLOT0:%[a-z0-9]+]] = llvm.getelementptr [[SLOTS]][0] : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    llvm.store [[T_ANY]], [[SLOT0]] : [[ANY]], !llvm.ptr
// CHECK-NEXT:    [[SLOT1:%[a-z0-9]+]] = llvm.getelementptr [[SLOTS]][1] : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    llvm.store [[B_ANY]], [[SLOT1]] : [[ANY]], !llvm.ptr
// --- Look up the ffi.Array handle and call it with the slot array.
// CHECK:         llvm.call @TVMFFIFunctionGetGlobal
// CHECK:         [[FUNC_HANDLE:%[a-z0-9]+]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @TVMFFIFunctionCall([[FUNC_HANDLE]], {{%.*}}, {{%.*}}, {{%.*}})
// CHECK-NEXT:    llvm.call @TVMFFIObjectDecRef([[FUNC_HANDLE]]) : (!llvm.ptr) -> i32
// --- Re-tag the container handle as a kTVMFFIArray (71) TVMFFIAny.
// CHECK:         llvm.getelementptr {{%.*}}[0, 2]
// CHECK-NEXT:    [[VOBJ:%[a-z0-9]+]] = llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK-NEXT:    [[UNDEF:%[a-z0-9]+]] = llvm.mlir.undef : [[ANY]]
// CHECK-NEXT:    %[[ARR_IDX:[0-9]+]] = llvm.mlir.constant(71 : i32) : i32
// CHECK-NEXT:    [[WITH_IDX:%[a-z0-9]+]] = llvm.insertvalue %[[ARR_IDX]], [[UNDEF]][0] : [[ANY]]
// CHECK-NEXT:    %[[ZERO32:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    [[WITH_PAD:%[a-z0-9]+]] = llvm.insertvalue %[[ZERO32]], [[WITH_IDX]][1] : [[ANY]]
// CHECK-NEXT:    [[ARRAY_ANY:%[a-z0-9]+]] = llvm.insertvalue [[VOBJ]], [[WITH_PAD]][2] : [[ANY]]
// CHECK-NEXT:    llvm.store [[ARRAY_ANY]], %arg3 : [[ANY]], !llvm.ptr
// --- DecRef the tensor element (the container owns its reference); the bool
//     is a scalar and must not appear here.
// CHECK-NEXT:    [[PAYLOAD:%[a-z0-9]+]] = llvm.extractvalue [[T_ANY]][2] : [[ANY]]
// CHECK-NEXT:    [[HANDLE:%[a-z0-9]+]] = llvm.inttoptr [[PAYLOAD]] : i64 to !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIObjectDecRef([[HANDLE]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @multi_return_tensor_bool(%arg0: !torch.tensor, %arg1: !torch.bool) -> (!torch.tensor, !torch.bool) {
  tvm_ffi.return %arg0, %arg1 : !torch.tensor, !torch.bool
}

// -----

// multi_return_int_int: scalar-only tuple.  No element DecRef may be emitted
// between the container store and the return — only the ffi.Array function
// handle DecRef inside the call helper.
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_int_int(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK:         %[[N:[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    [[SLOTS:%[a-z0-9]+]] = llvm.alloca %[[N]] x [[ANY]] : (i64) -> !llvm.ptr
// CHECK:         llvm.call @TVMFFIFunctionGetGlobal
// CHECK:         [[FUNC_HANDLE:%[a-z0-9]+]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @TVMFFIFunctionCall([[FUNC_HANDLE]], {{%.*}}, {{%.*}}, {{%.*}})
// CHECK-NEXT:    llvm.call @TVMFFIObjectDecRef([[FUNC_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK:         %[[ARR_IDX:[0-9]+]] = llvm.mlir.constant(71 : i32) : i32
// CHECK:         [[ARRAY_ANY:%[a-z0-9]+]] = llvm.insertvalue {{%.*}}, {{%.*}}[2] : [[ANY]]
// CHECK-NEXT:    llvm.store [[ARRAY_ANY]], %arg3 : [[ANY]], !llvm.ptr
// CHECK-NOT:     llvm.call @TVMFFIObjectDecRef
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @multi_return_int_int(%arg0: !torch.int, %arg1: !torch.int) -> (!torch.int, !torch.int) {
  tvm_ffi.return %arg0, %arg1 : !torch.int, !torch.int
}

// -----

// multi_return_three_tensors: three object elements → three element DecRefs
// after the container store.
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_three_tensors(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// --- Allocate three slots and fill each with a casted tensor.
// CHECK:         %[[N:[0-9]+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    [[SLOTS:%[a-z0-9]+]] = llvm.alloca %[[N]] x [[ANY]] : (i64) -> !llvm.ptr
// CHECK-NEXT:    [[SLOT0:%[a-z0-9]+]] = llvm.getelementptr [[SLOTS]][0] : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    llvm.store {{%.*}}, [[SLOT0]] : [[ANY]], !llvm.ptr
// CHECK-NEXT:    [[SLOT1:%[a-z0-9]+]] = llvm.getelementptr [[SLOTS]][1] : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    llvm.store {{%.*}}, [[SLOT1]] : [[ANY]], !llvm.ptr
// CHECK-NEXT:    [[SLOT2:%[a-z0-9]+]] = llvm.getelementptr [[SLOTS]][2] : (!llvm.ptr) -> !llvm.ptr, [[ANY]]
// CHECK-NEXT:    llvm.store {{%.*}}, [[SLOT2]] : [[ANY]], !llvm.ptr
// --- ffi.Array call helper + kTVMFFIArray re-tag + store into %arg3.
// CHECK:         llvm.call @TVMFFIFunctionGetGlobal
// CHECK:         [[FUNC_HANDLE:%[a-z0-9]+]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @TVMFFIFunctionCall([[FUNC_HANDLE]], {{%.*}}, {{%.*}}, {{%.*}})
// CHECK-NEXT:    llvm.call @TVMFFIObjectDecRef([[FUNC_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK:         %[[ARR_IDX:[0-9]+]] = llvm.mlir.constant(71 : i32) : i32
// CHECK:         [[ARRAY_ANY:%[a-z0-9]+]] = llvm.insertvalue {{%.*}}, {{%.*}}[2] : [[ANY]]
// CHECK-NEXT:    llvm.store [[ARRAY_ANY]], %arg3 : [[ANY]], !llvm.ptr
// --- Exactly three element DecRefs (one per tensor).
// CHECK-COUNT-3: llvm.call @TVMFFIObjectDecRef
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @multi_return_three_tensors(%arg0: !torch.tensor, %arg1: !torch.tensor, %arg2: !torch.tensor) -> (!torch.tensor, !torch.tensor, !torch.tensor) {
  tvm_ffi.return %arg0, %arg1, %arg2 : !torch.tensor, !torch.tensor, !torch.tensor
}

