//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file --trident-lowering-pipeline | FileCheck %s --check-prefix=STRICT
// RUN: trident-core-opt %s -split-input-file -convert-tvm-ffi-to-func | FileCheck %s --check-prefix=INTERMEDIATE
// RUN: trident-core-opt %s -split-input-file -convert-tvm-ffi-to-func | FileCheck %s --check-prefix=FUNC
// STRICT-NOT: !torch.
// STRICT: llvm.func @make_int() -> !llvm.struct<(i32, i32, i64)>
// STRICT: llvm.func @multi_return_three_tensors(

// === Section 1: no-result / single-result returns ===

// void_func: no results -> nothing is stored into %arg3; the function just
// returns 0.
// CHECK-LABEL: llvm.func @__tvm_ffi_void_func(
// CHECK: [[VOID_TRUE:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(true) : i1
// CHECK: llvm.cond_br [[VOID_TRUE]], [[VOID_SUCCESS:\^bb[0-9]+]], [[VOID_FAILURE:\^bb[0-9]+]]
// CHECK: [[VOID_SUCCESS]]:
// CHECK-NEXT: llvm.call @void_func() : () -> ()
// CHECK-NEXT: llvm.br [[VOID_MERGE:\^bb[0-9]+]]
// CHECK: [[VOID_MERGE]]:
// CHECK-NEXT: [[VOID_STATUS:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: llvm.return [[VOID_STATUS]] : i32
tvm_ffi.func @void_func() attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return
}

// -----

// make_int: a single scalar result is cast to its converted (TVMFFIAny)
// type and stored directly into %arg3.
// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK: [[MAKE_INT_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @make_int() : () -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[MAKE_INT_TORCH:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[MAKE_INT_RESULT]] : !llvm.struct<(i32, i32, i64)> to !torch.int
// CHECK-NEXT: [[MAKE_INT_ANY:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[MAKE_INT_TORCH]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[MAKE_INT_ANY]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @make_int() -> !torch.int attributes {emit_tvm_ffi_abi} {
  %0 = torch.constant.int 42
  tvm_ffi.return %0 : !torch.int
}

// -----

// dtype_constant: pack the DLPack dtype fields into the TVMFFI Any payload.
// CHECK-LABEL: llvm.func @dtype_constant(
// CHECK-NOT: !llvm.struct<packed (i8, i8, i16)>
// CHECK: [[DTYPE_PACKED:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(73730 : i64) : i64
tvm_ffi.func @dtype_constant() -> !tvm_ffi.dtype attributes {emit_tvm_ffi_abi} {
  %dtype = "tvm_ffi.constant"() <{value = [2, 32, 1]}> : () -> !tvm_ffi.dtype
  tvm_ffi.return %dtype : !tvm_ffi.dtype
}

// -----

// === Section 2: argument marshalling ===

// print_int: an !torch.int argument is unpacked from the arg array — GEP into
// the slot, load the TVMFFIAny, cast to the Torch type.
// CHECK-LABEL: llvm.func @__tvm_ffi_print_int(
// CHECK: [[PRINT_INT_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[PRINT_INT_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[PRINT_INT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[PRINT_INT_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[PRINT_INT_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[PRINT_INT_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: [[PRINT_INT_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[PRINT_INT_TYPE]], [[PRINT_INT_KIND]] : i32
// CHECK-NEXT: llvm.cond_br [[PRINT_INT_VALID]], [[PRINT_INT_SUCCESS:\^bb[0-9]+]], [[PRINT_INT_FAILURE:\^bb[0-9]+]]
// CHECK: [[PRINT_INT_SUCCESS]]:
// CHECK-NEXT: llvm.call @print_int([[PRINT_INT_ANY]]) : (!llvm.struct<(i32, i32, i64)>) -> ()
tvm_ffi.func @print_int(%arg0: !torch.int) attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return
}

// -----

// identity_bool: an !torch.bool argument is unpacked and returned unchanged.
// On return the Torch value is cast back to TVMFFIAny and stored into %arg3.
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_bool(
// CHECK: [[BOOL_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[BOOL_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[BOOL_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[BOOL_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[BOOL_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[BOOL_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: [[BOOL_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[BOOL_TYPE]], [[BOOL_KIND]] : i32
// CHECK-NEXT: llvm.cond_br [[BOOL_VALID]], [[BOOL_SUCCESS:\^bb[0-9]+]], [[BOOL_FAILURE:\^bb[0-9]+]]
// CHECK: [[BOOL_SUCCESS]]:
// CHECK-NEXT: [[BOOL_CALL_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @identity_bool([[BOOL_ANY]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[BOOL_TORCH_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[BOOL_CALL_RESULT]] : !llvm.struct<(i32, i32, i64)> to !torch.bool
// CHECK-NEXT: [[BOOL_ABI_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[BOOL_TORCH_RESULT]] : !torch.bool to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[BOOL_ABI_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @identity_bool(%arg0: !torch.bool) -> !torch.bool attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %arg0 : !torch.bool
}

// -----

// identity_float: same round-trip as identity_bool, for !torch.float.
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_float(
// CHECK: [[FLOAT_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[FLOAT_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[FLOAT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[FLOAT_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[FLOAT_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[FLOAT_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: [[FLOAT_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[FLOAT_TYPE]], [[FLOAT_KIND]] : i32
// CHECK-NEXT: llvm.cond_br [[FLOAT_VALID]], [[FLOAT_SUCCESS:\^bb[0-9]+]], [[FLOAT_FAILURE:\^bb[0-9]+]]
// CHECK: [[FLOAT_SUCCESS]]:
// CHECK-NEXT: [[FLOAT_CALL_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @identity_float([[FLOAT_ANY]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[FLOAT_TORCH_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[FLOAT_CALL_RESULT]] : !llvm.struct<(i32, i32, i64)> to !torch.float
// CHECK-NEXT: [[FLOAT_ABI_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[FLOAT_TORCH_RESULT]] : !torch.float to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[FLOAT_ABI_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @identity_float(%arg0: !torch.float) -> !torch.float attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %arg0 : !torch.float
}

// -----

// tensor_func: an !torch.tensor argument is unpacked; nothing is returned.
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_func(
// CHECK: [[TENSOR_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TENSOR_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[TENSOR_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[TENSOR_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TENSOR_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TENSOR_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(70 : i32) : i32
// CHECK-NEXT: [[TENSOR_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[TENSOR_TYPE]], [[TENSOR_KIND]] : i32
// CHECK-NEXT: llvm.cond_br [[TENSOR_VALID]], [[TENSOR_SUCCESS:\^bb[0-9]+]], [[TENSOR_FAILURE:\^bb[0-9]+]]
// CHECK: [[TENSOR_SUCCESS]]:
// CHECK-NEXT: llvm.call @tensor_func([[TENSOR_ANY]]) : (!llvm.struct<(i32, i32, i64)>) -> ()
tvm_ffi.func @tensor_func(%arg0: !torch.tensor) attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return
}

// -----

// make_tensor: a tensor produced by torch.aten.empty.memory_format is
// returned; the single TVMFFIAny result is stored into %arg3.
// CHECK-LABEL: llvm.func @__tvm_ffi_make_tensor(
// CHECK: [[MAKE_TENSOR_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @make_tensor() : () -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[MAKE_TENSOR_TORCH:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[MAKE_TENSOR_RESULT]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK-NEXT: [[MAKE_TENSOR_ANY:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[MAKE_TENSOR_TORCH]] : !torch.tensor to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[MAKE_TENSOR_ANY]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @make_tensor() -> !torch.tensor attributes {emit_tvm_ffi_abi} {
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
// seqlenq_ngroups_swapped scenario.
// * Cast each result to its converted TVMFFIAny type.
// * Allocate the per-result slot array and fill it.
// * Look up the ffi.Array handle and call it with the slot array.
// * Re-tag the container handle as a kTVMFFIArray (71) TVMFFIAny.
// * Release only the temporary ffi.Array function handle; returned inputs are
//   borrowed and neither the tensor nor the scalar element is DecRef'd.
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_tensor_bool(
// CHECK: [[TUPLE_TENSOR_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TUPLE_TENSOR:%[a-zA-Z0-9_]+]] = llvm.load [[TUPLE_TENSOR_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[TUPLE_TENSOR_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TUPLE_TENSOR]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TUPLE_TENSOR_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(70 : i32) : i32
// CHECK-NEXT: [[TUPLE_TENSOR_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[TUPLE_TENSOR_TYPE]], [[TUPLE_TENSOR_KIND]] : i32
// CHECK-NEXT: [[TUPLE_BOOL_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TUPLE_BOOL:%[a-zA-Z0-9_]+]] = llvm.load [[TUPLE_BOOL_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[TUPLE_BOOL_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TUPLE_BOOL]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TUPLE_BOOL_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: [[TUPLE_BOOL_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[TUPLE_BOOL_TYPE]], [[TUPLE_BOOL_KIND]] : i32
// CHECK-NEXT: [[TUPLE_ARGS_VALID:%[a-zA-Z0-9_]+]] = llvm.and [[TUPLE_TENSOR_VALID]], [[TUPLE_BOOL_VALID]] : i1
// CHECK-NEXT: llvm.cond_br [[TUPLE_ARGS_VALID]], [[TUPLE_SUCCESS:\^bb[0-9]+]], [[TUPLE_FAILURE:\^bb[0-9]+]]
// CHECK: [[TUPLE_SUCCESS]]:
// CHECK-NEXT: [[TUPLE_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @multi_return_tensor_bool([[TUPLE_TENSOR]], [[TUPLE_BOOL]])
// CHECK-NEXT: [[TUPLE_RESULT0:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TUPLE_RESULT]][0]
// CHECK-NEXT: [[TUPLE_RESULT1:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TUPLE_RESULT]][1]
// CHECK-NEXT: [[TUPLE_BOOL_TORCH:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[TUPLE_RESULT1]] : !llvm.struct<(i32, i32, i64)> to !torch.bool
// CHECK-NEXT: [[TUPLE_TENSOR_TORCH:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[TUPLE_RESULT0]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK: [[TUPLE_ELEMENT_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[TUPLE_ELEMENT_SLOTS:%[a-zA-Z0-9_]+]] = llvm.alloca [[TUPLE_ELEMENT_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: [[TUPLE_TENSOR_ANY:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[TUPLE_TENSOR_TORCH]] : !torch.tensor to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TUPLE_TENSOR_RESULT_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[TUPLE_ELEMENT_SLOTS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[TUPLE_TENSOR_ANY]], [[TUPLE_TENSOR_RESULT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[TUPLE_BOOL_ANY:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[TUPLE_BOOL_TORCH]] : !torch.bool to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TUPLE_BOOL_RESULT_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[TUPLE_ELEMENT_SLOTS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[TUPLE_BOOL_ANY]], [[TUPLE_BOOL_RESULT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[TUPLE_CALL_ARG_COUNT64:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[TUPLE_CALL_ARGS:%[a-zA-Z0-9_]+]] = llvm.alloca [[TUPLE_CALL_ARG_COUNT64]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: [[TUPLE_CALL_ARG0:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[TUPLE_CALL_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TUPLE_ARG0_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[TUPLE_TENSOR_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[TUPLE_ARG0_COPY]], [[TUPLE_CALL_ARG0]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-NEXT: [[TUPLE_CALL_ARG1:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[TUPLE_CALL_ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[TUPLE_ARG1_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[TUPLE_BOOL_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[TUPLE_ARG1_COPY]], [[TUPLE_CALL_ARG1]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[TUPLE_ARRAY_ARG_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: [[TUPLE_ARRAY_NAME:%[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_constant_ffi.Array_ffi.Array : !llvm.ptr
// CHECK: [[TUPLE_NAME_RECORD_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[TUPLE_NAME_BUFFER:%[a-zA-Z0-9_]+]] = llvm.alloca [[TUPLE_NAME_RECORD_COUNT]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// CHECK: [[TUPLE_NAME_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[TUPLE_NAME_BUFFER]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK-NEXT: llvm.store [[TUPLE_ARRAY_NAME]], [[TUPLE_NAME_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: [[TUPLE_NAME_LENGTH:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(9 : i64) : i64
// CHECK-NEXT: [[TUPLE_NAME_LENGTH_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[TUPLE_NAME_BUFFER]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK-NEXT: llvm.store [[TUPLE_NAME_LENGTH]], [[TUPLE_NAME_LENGTH_PTR]] : i64, !llvm.ptr
// CHECK: [[TUPLE_HANDLE_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[TUPLE_ARRAY_HANDLE_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[TUPLE_HANDLE_COUNT]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT: [[TUPLE_GET_GLOBAL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal([[TUPLE_NAME_BUFFER]], [[TUPLE_ARRAY_HANDLE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: [[TUPLE_ARRAY_HANDLE:%[a-zA-Z0-9_]+]] = llvm.load [[TUPLE_ARRAY_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: [[TUPLE_RETURN_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[TUPLE_ARRAY_RETURN_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[TUPLE_RETURN_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: [[TUPLE_CALL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall([[TUPLE_ARRAY_HANDLE]], [[TUPLE_CALL_ARGS]], [[TUPLE_ARRAY_ARG_COUNT]], [[TUPLE_ARRAY_RETURN_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-NEXT: [[TUPLE_DEC_REF_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIObjectDecRef([[TUPLE_ARRAY_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK-NEXT: [[TUPLE_ARRAY_RESULT:%[a-zA-Z0-9_]+]] = llvm.load [[TUPLE_ARRAY_RETURN_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[TUPLE_ARRAY_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @multi_return_tensor_bool(%arg0: !torch.tensor, %arg1: !torch.bool) -> (!torch.tensor, !torch.bool) attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %arg0, %arg1 : !torch.tensor, !torch.bool
}

// -----

// multi_return_int_int: scalar-only tuple.  No element DecRef may be emitted
// between the container store and the return — only the ffi.Array function
// handle DecRef inside the call helper.
// CHECK-LABEL: llvm.func @__tvm_ffi_multi_return_int_int(
// CHECK: [[INT0_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[INT0_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[INT0_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[INT0_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[INT0_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[INT0_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: [[INT0_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[INT0_TYPE]], [[INT0_KIND]] : i32
// CHECK-NEXT: [[INT1_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[INT1_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[INT1_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[INT1_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[INT1_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[INT1_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: [[INT1_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[INT1_TYPE]], [[INT1_KIND]] : i32
// CHECK-NEXT: [[INT_ARGS_VALID:%[a-zA-Z0-9_]+]] = llvm.and [[INT0_VALID]], [[INT1_VALID]] : i1
// CHECK-NEXT: llvm.cond_br [[INT_ARGS_VALID]], [[INT_TUPLE_SUCCESS:\^bb[0-9]+]], [[INT_TUPLE_FAILURE:\^bb[0-9]+]]
// CHECK: [[INT_TUPLE_SUCCESS]]:
// CHECK-NEXT: [[INT_TUPLE_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @multi_return_int_int([[INT0_ANY]], [[INT1_ANY]])
// CHECK-NEXT: [[INT_TUPLE_RESULT0:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[INT_TUPLE_RESULT]][0]
// CHECK-NEXT: [[INT_TUPLE_RESULT1:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[INT_TUPLE_RESULT]][1]
// CHECK-NEXT: [[INT1_TORCH:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[INT_TUPLE_RESULT1]] : !llvm.struct<(i32, i32, i64)> to !torch.int
// CHECK-NEXT: [[INT0_TORCH:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[INT_TUPLE_RESULT0]] : !llvm.struct<(i32, i32, i64)> to !torch.int
// CHECK: [[INT_ELEMENT_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[INT_ELEMENT_SLOTS:%[a-zA-Z0-9_]+]] = llvm.alloca [[INT_ELEMENT_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK-NEXT: [[INT0_RESULT_ANY:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[INT0_TORCH]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[INT0_RESULT_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[INT_ELEMENT_SLOTS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[INT0_RESULT_ANY]], [[INT0_RESULT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-NEXT: [[INT1_RESULT_ANY:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[INT1_TORCH]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[INT1_RESULT_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[INT_ELEMENT_SLOTS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[INT1_RESULT_ANY]], [[INT1_RESULT_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[INT_CALL_ARG_COUNT64:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: [[INT_CALL_ARGS:%[a-zA-Z0-9_]+]] = llvm.alloca [[INT_CALL_ARG_COUNT64]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: [[INT_CALL_ARG0:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[INT_CALL_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[INT_ARG0_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[INT0_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[INT_ARG0_COPY]], [[INT_CALL_ARG0]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-NEXT: [[INT_CALL_ARG1:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[INT_CALL_ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[INT_ARG1_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[INT1_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[INT_ARG1_COPY]], [[INT_CALL_ARG1]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[INT_ARRAY_ARG_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: [[INT_ARRAY_NAME:%[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_constant_ffi.Array_ffi.Array : !llvm.ptr
// CHECK: [[INT_NAME_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[INT_NAME_BUFFER:%[a-zA-Z0-9_]+]] = llvm.alloca [[INT_NAME_COUNT]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// CHECK: [[INT_NAME_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[INT_NAME_BUFFER]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK-NEXT: llvm.store [[INT_ARRAY_NAME]], [[INT_NAME_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: [[INT_HANDLE_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[INT_ARRAY_HANDLE_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[INT_HANDLE_COUNT]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT: [[INT_GET_GLOBAL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal([[INT_NAME_BUFFER]], [[INT_ARRAY_HANDLE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: [[INT_ARRAY_HANDLE:%[a-zA-Z0-9_]+]] = llvm.load [[INT_ARRAY_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: [[INT_RETURN_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[INT_ARRAY_RETURN_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[INT_RETURN_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: [[INT_ARRAY_CALL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall([[INT_ARRAY_HANDLE]], [[INT_CALL_ARGS]], [[INT_ARRAY_ARG_COUNT]], [[INT_ARRAY_RETURN_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-NEXT: [[INT_ARRAY_DEC_REF:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIObjectDecRef([[INT_ARRAY_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK-NEXT: [[INT_ARRAY_RESULT:%[a-zA-Z0-9_]+]] = llvm.load [[INT_ARRAY_RETURN_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[INT_ARRAY_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @multi_return_int_int(%arg0: !torch.int, %arg1: !torch.int) -> (!torch.int, !torch.int) attributes {emit_tvm_ffi_abi} {
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
// CHECK: [[THREE_TENSOR0_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_TENSOR0:%[a-zA-Z0-9_]+]] = llvm.load [[THREE_TENSOR0_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[THREE_TENSOR0_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[THREE_TENSOR0]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_TENSOR0_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(70 : i32) : i32
// CHECK-NEXT: [[THREE_TENSOR0_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[THREE_TENSOR0_TYPE]], [[THREE_TENSOR0_KIND]] : i32
// CHECK-NEXT: [[THREE_TENSOR1_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_TENSOR1:%[a-zA-Z0-9_]+]] = llvm.load [[THREE_TENSOR1_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[THREE_TENSOR1_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[THREE_TENSOR1]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_TENSOR1_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(70 : i32) : i32
// CHECK-NEXT: [[THREE_TENSOR1_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[THREE_TENSOR1_TYPE]], [[THREE_TENSOR1_KIND]] : i32
// CHECK-NEXT: [[FIRST_TWO_TENSORS_VALID:%[a-zA-Z0-9_]+]] = llvm.and [[THREE_TENSOR0_VALID]], [[THREE_TENSOR1_VALID]] : i1
// CHECK-NEXT: [[THREE_TENSOR2_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_TENSOR2:%[a-zA-Z0-9_]+]] = llvm.load [[THREE_TENSOR2_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[THREE_TENSOR2_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[THREE_TENSOR2]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_TENSOR2_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(70 : i32) : i32
// CHECK-NEXT: [[THREE_TENSOR2_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[THREE_TENSOR2_TYPE]], [[THREE_TENSOR2_KIND]] : i32
// CHECK-NEXT: [[THREE_TENSORS_VALID:%[a-zA-Z0-9_]+]] = llvm.and [[FIRST_TWO_TENSORS_VALID]], [[THREE_TENSOR2_VALID]] : i1
// CHECK-NEXT: llvm.cond_br [[THREE_TENSORS_VALID]], [[THREE_SUCCESS:\^bb[0-9]+]], [[THREE_FAILURE:\^bb[0-9]+]]
// CHECK: [[THREE_SUCCESS]]:
// CHECK-NEXT: [[THREE_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @multi_return_three_tensors([[THREE_TENSOR0]], [[THREE_TENSOR1]], [[THREE_TENSOR2]])
// CHECK-NEXT: [[THREE_RESULT0:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[THREE_RESULT]][0]
// CHECK-NEXT: [[THREE_RESULT1:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[THREE_RESULT]][1]
// CHECK-NEXT: [[THREE_RESULT2:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[THREE_RESULT]][2]
// CHECK-NEXT: [[THREE_TORCH2:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[THREE_RESULT2]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK-NEXT: [[THREE_TORCH1:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[THREE_RESULT1]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK-NEXT: [[THREE_TORCH0:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[THREE_RESULT0]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK: [[THREE_ELEMENT_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: [[THREE_ELEMENT_SLOTS:%[a-zA-Z0-9_]+]] = llvm.alloca [[THREE_ELEMENT_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: [[THREE_ANY0:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[THREE_TORCH0]] : !torch.tensor to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_SLOT0:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[THREE_ELEMENT_SLOTS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[THREE_ANY0]], [[THREE_SLOT0]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[THREE_ANY1:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[THREE_TORCH1]] : !torch.tensor to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_SLOT1:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[THREE_ELEMENT_SLOTS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[THREE_ANY1]], [[THREE_SLOT1]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[THREE_ANY2:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[THREE_TORCH2]] : !torch.tensor to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_SLOT2:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[THREE_ELEMENT_SLOTS]][2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[THREE_ANY2]], [[THREE_SLOT2]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[THREE_CALL_ARG_COUNT64:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT: [[THREE_CALL_ARGS:%[a-zA-Z0-9_]+]] = llvm.alloca [[THREE_CALL_ARG_COUNT64]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK-NEXT: [[THREE_CALL_ARG0:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[THREE_CALL_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_ARG0_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[THREE_SLOT0]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[THREE_ARG0_COPY]], [[THREE_CALL_ARG0]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-NEXT: [[THREE_CALL_ARG1:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[THREE_CALL_ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_ARG1_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[THREE_SLOT1]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[THREE_ARG1_COPY]], [[THREE_CALL_ARG1]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-NEXT: [[THREE_CALL_ARG2:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[THREE_CALL_ARGS]][2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[THREE_ARG2_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[THREE_SLOT2]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[THREE_ARG2_COPY]], [[THREE_CALL_ARG2]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[THREE_ARRAY_ARG_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: [[THREE_ARRAY_NAME:%[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_constant_ffi.Array_ffi.Array : !llvm.ptr
// CHECK: [[THREE_NAME_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[THREE_NAME_BUFFER:%[a-zA-Z0-9_]+]] = llvm.alloca [[THREE_NAME_COUNT]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// CHECK: [[THREE_NAME_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[THREE_NAME_BUFFER]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// CHECK-NEXT: llvm.store [[THREE_ARRAY_NAME]], [[THREE_NAME_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: [[THREE_HANDLE_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[THREE_ARRAY_HANDLE_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[THREE_HANDLE_COUNT]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT: [[THREE_GET_GLOBAL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal([[THREE_NAME_BUFFER]], [[THREE_ARRAY_HANDLE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT: [[THREE_ARRAY_HANDLE:%[a-zA-Z0-9_]+]] = llvm.load [[THREE_ARRAY_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: [[THREE_RETURN_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[THREE_ARRAY_RETURN_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[THREE_RETURN_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: [[THREE_CALL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall([[THREE_ARRAY_HANDLE]], [[THREE_CALL_ARGS]], [[THREE_ARRAY_ARG_COUNT]], [[THREE_ARRAY_RETURN_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-NEXT: [[THREE_DEC_REF_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIObjectDecRef([[THREE_ARRAY_HANDLE]]) : (!llvm.ptr) -> i32
// CHECK-NEXT: [[THREE_ARRAY_RESULT:%[a-zA-Z0-9_]+]] = llvm.load [[THREE_ARRAY_RETURN_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[THREE_ARRAY_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @multi_return_three_tensors(%arg0: !torch.tensor, %arg1: !torch.tensor, %arg2: !torch.tensor) -> (!torch.tensor, !torch.tensor, !torch.tensor) attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %arg0, %arg1, %arg2 : !torch.tensor, !torch.tensor, !torch.tensor
}

// -----

// Device constants use Torch's device parser and preserve the complete
// DLPack device value in the TVM FFI Any payload.
// CHECK-LABEL: llvm.func @constant_cpu_device(
// CHECK: [[CPU_DEVICE_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
// CHECK-NEXT: [[CPU_DEVICE_TYPE:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: [[CPU_WITH_TYPE:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CPU_DEVICE_TYPE]], [[CPU_DEVICE_UNDEF]][0] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: [[CPU_DEVICE_INDEX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: [[CPU_DEVICE:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CPU_DEVICE_INDEX]], [[CPU_WITH_TYPE]][1] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: [[CPU_DEVICE_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[CPU_DEVICE_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[CPU_DEVICE_COUNT]] x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT: llvm.store [[CPU_DEVICE]], [[CPU_DEVICE_SLOT]] : !llvm.struct<(i32, i32)>, !llvm.ptr
// CHECK-NEXT: [[CPU_DEVICE_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.load [[CPU_DEVICE_SLOT]] : !llvm.ptr -> i64
// CHECK: [[CPU_ANY_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK: [[CPU_ANY_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(6 : i32) : i32
// CHECK: [[CPU_ANY_WITH_KIND:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CPU_ANY_KIND]], [[CPU_ANY_UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK: [[CPU_ANY_AUX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[CPU_ANY_WITH_AUX:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CPU_ANY_AUX]], [[CPU_ANY_WITH_KIND]][1] : !llvm.struct<(i32, i32, i64)>
// CHECK: [[CPU_ANY:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CPU_DEVICE_PAYLOAD]], [[CPU_ANY_WITH_AUX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.return [[CPU_ANY]] : !llvm.struct<(i32, i32, i64)>
tvm_ffi.func @constant_cpu_device() -> !tvm_ffi.device attributes {emit_tvm_ffi_abi} {
  %0 = "tvm_ffi.constant"() <{value = "cpu"}> : () -> !tvm_ffi.device
  tvm_ffi.return %0 : !tvm_ffi.device
}

// -----

// CHECK-LABEL: llvm.func @constant_cuda_device(
// CHECK: [[CUDA_DEVICE_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
// CHECK-NEXT: [[CUDA_DEVICE_TYPE:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: [[CUDA_WITH_TYPE:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CUDA_DEVICE_TYPE]], [[CUDA_DEVICE_UNDEF]][0] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: [[CUDA_DEVICE_INDEX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: [[CUDA_DEVICE:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CUDA_DEVICE_INDEX]], [[CUDA_WITH_TYPE]][1] : !llvm.struct<(i32, i32)>
// CHECK-NEXT: [[CUDA_DEVICE_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[CUDA_DEVICE_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[CUDA_DEVICE_COUNT]] x !llvm.struct<(i32, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT: llvm.store [[CUDA_DEVICE]], [[CUDA_DEVICE_SLOT]] : !llvm.struct<(i32, i32)>, !llvm.ptr
// CHECK-NEXT: [[CUDA_DEVICE_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.load [[CUDA_DEVICE_SLOT]] : !llvm.ptr -> i64
// CHECK: [[CUDA_ANY_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK: [[CUDA_ANY_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(6 : i32) : i32
// CHECK: [[CUDA_ANY_WITH_KIND:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CUDA_ANY_KIND]], [[CUDA_ANY_UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK: [[CUDA_ANY_AUX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[CUDA_ANY_WITH_AUX:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CUDA_ANY_AUX]], [[CUDA_ANY_WITH_KIND]][1] : !llvm.struct<(i32, i32, i64)>
// CHECK: [[CUDA_ANY:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[CUDA_DEVICE_PAYLOAD]], [[CUDA_ANY_WITH_AUX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.return [[CUDA_ANY]] : !llvm.struct<(i32, i32, i64)>
tvm_ffi.func @constant_cuda_device() -> !tvm_ffi.device attributes {emit_tvm_ffi_abi} {
  %0 = "tvm_ffi.constant"() <{value = "cuda:1"}> : () -> !tvm_ffi.device
  tvm_ffi.return %0 : !tvm_ffi.device
}

// -----

// A semantic call to a local TVMFFI function is lowered to an ABI-level
// func.call. The later FuncToLLVM conversion turns this into llvm.call.
// CHECK-LABEL: llvm.func @callee(
// CHECK-SAME: [[CALLEE_ARG:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK-NEXT: llvm.return [[CALLEE_ARG]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_callee(
// CHECK: [[CALLEE_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[CALLEE_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[CALLEE_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[CALLEE_CALL_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @callee([[CALLEE_ANY]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[CALLEE_TVM_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[CALLEE_CALL_RESULT]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.int
// CHECK-NEXT: [[CALLEE_ABI_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[CALLEE_TVM_RESULT]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[CALLEE_ABI_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK-LABEL: llvm.func @caller(
// CHECK-SAME: [[CALLER_ARG:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: [[CALLER_ARG_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: [[CALLER_ARG_COUNT64:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[CALLER_ARGS:%[a-zA-Z0-9_]+]] = llvm.alloca [[CALLER_ARG_COUNT64]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK-NEXT: [[CALLER_ARG0:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[CALLER_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[CALLER_ARG]], [[CALLER_ARG0]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK: [[CALLER_RESULT_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT: [[CALLER_RESULT_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[CALLER_RESULT_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// CHECK: [[CALLER_CONTEXT:%[a-zA-Z0-9_]+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: [[CALLER_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @__tvm_ffi_callee([[CALLER_CONTEXT]], [[CALLER_ARGS]], [[CALLER_ARG_COUNT]], [[CALLER_RESULT_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-NEXT: [[CALLER_RESULT:%[a-zA-Z0-9_]+]] = llvm.load [[CALLER_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.return [[CALLER_RESULT]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_caller(
// CHECK: [[CALLER_WRAPPER_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[CALLER_WRAPPER_ARG:%[a-zA-Z0-9_]+]] = llvm.load [[CALLER_WRAPPER_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: [[CALLER_WRAPPER_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @caller([[CALLER_WRAPPER_ARG]]) : (!llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: [[CALLER_WRAPPER_TVM_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[CALLER_WRAPPER_RESULT]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.int
// CHECK-NEXT: [[CALLER_WRAPPER_ABI_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[CALLER_WRAPPER_TVM_RESULT]] : !tvm_ffi.int to !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT: llvm.store [[CALLER_WRAPPER_ABI_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @callee(%arg0: !tvm_ffi.int) -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %arg0 : !tvm_ffi.int
}

tvm_ffi.func @caller(%arg0: !tvm_ffi.int) -> !tvm_ffi.int attributes {emit_tvm_ffi_abi} {
  %0 = tvm_ffi.call @callee(%arg0) : (!tvm_ffi.int) -> !tvm_ffi.int
  tvm_ffi.return %0 : !tvm_ffi.int
}

// -----

// TVMFFIToFunc preserves CFG block argument types and forwards values directly.
// The unified LLVM conversion later converts both the branch and block.
// INTERMEDIATE-LABEL: func.func @branch_argument(
// INTERMEDIATE: [[BRANCH_VALUE:%[a-zA-Z0-9_]+]] = torch.constant.int 7
// INTERMEDIATE-NEXT: cf.br [[BRANCH_DEST:\^bb[0-9]+]]([[BRANCH_VALUE]] : !torch.int)
// INTERMEDIATE: [[BRANCH_DEST]]([[BRANCH_BLOCK_ARG:%[a-zA-Z0-9_]+]]: !torch.int):
// INTERMEDIATE-NEXT: return [[BRANCH_BLOCK_ARG]] : !torch.int
// FINAL-LABEL: llvm.func @branch_argument(
// FINAL: [[BRANCH_AUX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// FINAL-NEXT: [[BRANCH_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(7 : i64) : i64
// FINAL-NEXT: [[BRANCH_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[BRANCH_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// FINAL-NEXT: [[BRANCH_WITH_KIND:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[BRANCH_KIND]], [[BRANCH_UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[BRANCH_WITH_AUX:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[BRANCH_AUX]], [[BRANCH_WITH_KIND]][1] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[BRANCH_ANY:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[BRANCH_PAYLOAD]], [[BRANCH_WITH_AUX]][2] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: llvm.br [[LLVM_BRANCH_DEST:\^bb[0-9]+]]
// FINAL: [[LLVM_BRANCH_DEST]]:
// FINAL-NEXT: llvm.return [[BRANCH_ANY]] : !llvm.struct<(i32, i32, i64)>
tvm_ffi.func @branch_argument() -> !torch.int attributes {emit_tvm_ffi_abi} {
  %value = torch.constant.int 7
  cf.br ^bb1(%value : !torch.int)

^bb1(%arg: !torch.int):
  tvm_ffi.return %arg : !torch.int
}

// -----

// Both edges of a conditional branch carry their own destination operands.
// INTERMEDIATE-LABEL: func.func @cond_branch_arguments(
// INTERMEDIATE: [[COND:%[a-zA-Z0-9_]+]] = arith.constant true
// INTERMEDIATE-NEXT: [[TRUE_VALUE:%[a-zA-Z0-9_]+]] = torch.constant.int 1
// INTERMEDIATE-NEXT: [[FALSE_VALUE:%[a-zA-Z0-9_]+]] = torch.constant.int 2
// INTERMEDIATE-NEXT: cf.cond_br [[COND]] weights([90, 10]), [[TRUE_DEST:\^bb[0-9]+]]([[TRUE_VALUE]] : !torch.int), [[FALSE_DEST:\^bb[0-9]+]]([[FALSE_VALUE]] : !torch.int)
// INTERMEDIATE: [[TRUE_DEST]]([[TRUE_BLOCK_ARG:%[a-zA-Z0-9_]+]]: !torch.int):
// INTERMEDIATE-NEXT: return [[TRUE_BLOCK_ARG]] : !torch.int
// INTERMEDIATE: [[FALSE_DEST]]([[FALSE_BLOCK_ARG:%[a-zA-Z0-9_]+]]: !torch.int):
// INTERMEDIATE-NEXT: return [[FALSE_BLOCK_ARG]] : !torch.int
// FINAL-LABEL: llvm.func @cond_branch_arguments(
// FINAL: [[FALSE_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i64) : i64
// FINAL-NEXT: [[COND_AUX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// FINAL-NEXT: [[LLVM_COND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(true) : i1
// FINAL-NEXT: [[TRUE_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// FINAL-NEXT: [[COND_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[COND_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// FINAL-NEXT: [[TRUE_WITH_KIND:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[COND_KIND]], [[COND_UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[TRUE_WITH_AUX:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[COND_AUX]], [[TRUE_WITH_KIND]][1] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[TRUE_ANY:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[TRUE_PAYLOAD]], [[TRUE_WITH_AUX]][2] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[FALSE_WITH_KIND:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[COND_KIND]], [[COND_UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[FALSE_WITH_AUX:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[COND_AUX]], [[FALSE_WITH_KIND]][1] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[FALSE_ANY:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[FALSE_PAYLOAD]], [[FALSE_WITH_AUX]][2] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: llvm.cond_br [[LLVM_COND]] weights([90, 10]), [[COND_MERGE:\^bb[0-9]+]]([[TRUE_ANY]] : !llvm.struct<(i32, i32, i64)>), [[COND_MERGE]]([[FALSE_ANY]] : !llvm.struct<(i32, i32, i64)>)
// FINAL: [[COND_MERGE]]([[COND_RESULT:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>):
// FINAL-NEXT: llvm.return [[COND_RESULT]] : !llvm.struct<(i32, i32, i64)>
tvm_ffi.func @cond_branch_arguments() -> !torch.int attributes {emit_tvm_ffi_abi} {
  %cond = arith.constant true
  %true_value = torch.constant.int 1
  %false_value = torch.constant.int 2
  cf.cond_br %cond, ^true(%true_value : !torch.int),
                         ^false(%false_value : !torch.int) {branch_weights = array<i32: 90, 10>}

^true(%true_arg: !torch.int):
  tvm_ffi.return %true_arg : !torch.int

^false(%false_arg: !torch.int):
  tvm_ffi.return %false_arg : !torch.int
}

// -----

// Switch destinations and forwarded operands are lowered by the standard CF
// interface using the shared LLVM type converter.
// INTERMEDIATE-LABEL: func.func @switch_argument(
// INTERMEDIATE-SAME: [[SWITCH_TENSOR:%[a-zA-Z0-9_]+]]: !torch.tensor, [[DEFAULT_VALUE:%[a-zA-Z0-9_]+]]: !torch.int, [[CASE_VALUE:%[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// INTERMEDIATE: [[SELECTOR:%[a-zA-Z0-9_]+]], [[DEVICE_INDEX:%[a-zA-Z0-9_]+]] = tvm_ffi.tensor.device [[SWITCH_TENSOR]] : !torch.tensor
// INTERMEDIATE-NEXT: cf.switch [[SELECTOR]] : i32, [
// INTERMEDIATE-NEXT: default: [[SWITCH_MERGE:\^bb[0-9]+]]([[DEFAULT_VALUE]] : !torch.int),
// INTERMEDIATE-NEXT: 1: [[SWITCH_MERGE]]([[CASE_VALUE]] : !torch.int)
// INTERMEDIATE-NEXT: ]
// INTERMEDIATE: [[SWITCH_MERGE]]([[SWITCH_RESULT:%[a-zA-Z0-9_]+]]: !torch.int):
// INTERMEDIATE-NEXT: return [[SWITCH_RESULT]] : !torch.int
// FINAL-LABEL: llvm.func @switch_argument(
// FINAL-SAME: [[LLVM_SWITCH_TENSOR:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, [[LLVM_DEFAULT:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, [[LLVM_CASE:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// FINAL: [[SWITCH_HANDLE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[LLVM_SWITCH_TENSOR]][2] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[SWITCH_OBJECT:%[a-zA-Z0-9_]+]] = llvm.inttoptr [[SWITCH_HANDLE]] : i64 to !llvm.ptr
// FINAL-NEXT: [[SWITCH_TENSOR_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[SWITCH_OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// FINAL-NEXT: [[SWITCH_DEVICE_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[SWITCH_TENSOR_PTR]][0, 1]
// FINAL-NEXT: [[SWITCH_DEVICE_TYPE_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[SWITCH_DEVICE_PTR]][0, 0]
// FINAL-NEXT: [[LLVM_SELECTOR:%[a-zA-Z0-9_]+]] = llvm.load [[SWITCH_DEVICE_TYPE_PTR]] : !llvm.ptr -> i32
// FINAL-NEXT: llvm.switch [[LLVM_SELECTOR]] : i32, [[LLVM_SWITCH_MERGE:\^bb[0-9]+]]([[LLVM_DEFAULT]] : !llvm.struct<(i32, i32, i64)>) [
// FINAL-NEXT: 1: [[LLVM_SWITCH_MERGE]]([[LLVM_CASE]] : !llvm.struct<(i32, i32, i64)>)
// FINAL-NEXT: ]
// FINAL: [[LLVM_SWITCH_MERGE]]([[LLVM_SWITCH_RESULT:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>):
// FINAL-NEXT: llvm.return [[LLVM_SWITCH_RESULT]] : !llvm.struct<(i32, i32, i64)>
tvm_ffi.func @switch_argument(
    %tensor: !torch.tensor,
    %default_value: !torch.int,
    %case_value: !torch.int) -> !torch.int {
  %selector, %device_index = tvm_ffi.tensor.device %tensor : !torch.tensor
  cf.switch %selector : i32, [
    default: ^merge(%default_value : !torch.int),
    1: ^merge(%case_value : !torch.int)
  ]

^merge(%value: !torch.int):
  tvm_ffi.return %value : !torch.int
}

// -----

// A wrapper is opt-in. Without the attribute, only the original function
// converted to func.func is generated.
// FUNC-LABEL: func.func @no_wrapper
// FUNC-NOT: func.func @__tvm_ffi_no_wrapper
// FUNC: [[NO_WRAPPER_VALUE:%[a-zA-Z0-9_]+]] = torch.constant.int 3
// FUNC-NEXT: return [[NO_WRAPPER_VALUE]] : !torch.int
// FINAL-LABEL: llvm.func @no_wrapper
// FINAL-NOT: llvm.func @__tvm_ffi_no_wrapper
// FINAL: [[NO_WRAPPER_AUX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// FINAL-NEXT: [[NO_WRAPPER_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(3 : i64) : i64
// FINAL-NEXT: [[NO_WRAPPER_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[NO_WRAPPER_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// FINAL-NEXT: [[NO_WRAPPER_WITH_KIND:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[NO_WRAPPER_KIND]], [[NO_WRAPPER_UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[NO_WRAPPER_WITH_AUX:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[NO_WRAPPER_AUX]], [[NO_WRAPPER_WITH_KIND]][1] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[NO_WRAPPER_ANY:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[NO_WRAPPER_PAYLOAD]], [[NO_WRAPPER_WITH_AUX]][2] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: llvm.return [[NO_WRAPPER_ANY]] : !llvm.struct<(i32, i32, i64)>
tvm_ffi.func @no_wrapper() -> !torch.int {
  %value = torch.constant.int 3
  tvm_ffi.return %value : !torch.int
}

// -----

// Guard failures are represented as nested SCF regions.  The final pipeline
// lowers those regions to CFG before converting to LLVM.
// INTERMEDIATE-LABEL: func.func @__tvm_ffi_guarded(
// INTERMEDIATE: [[GUARD_TRUE:%[a-zA-Z0-9_]+]] = arith.constant true
// INTERMEDIATE-NEXT: [[GUARD_INT_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_INT_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[GUARD_INT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_INT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[GUARD_INT_ANY]] : !llvm.struct<(i32, i32, i64)> to !torch.int
// INTERMEDIATE-NEXT: [[GUARD_INT_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[GUARD_INT_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_INT_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// INTERMEDIATE-NEXT: [[GUARD_INT_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[GUARD_INT_TYPE]], [[GUARD_INT_KIND]] : i32
// INTERMEDIATE-NEXT: [[GUARD_AFTER_INT:%[a-zA-Z0-9_]+]] = arith.andi [[GUARD_TRUE]], [[GUARD_INT_VALID]] : i1
// INTERMEDIATE-NEXT: [[GUARD_BOOL_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_BOOL_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[GUARD_BOOL_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_BOOL:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[GUARD_BOOL_ANY]] : !llvm.struct<(i32, i32, i64)> to !torch.bool
// INTERMEDIATE-NEXT: [[GUARD_BOOL_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[GUARD_BOOL_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: [[GUARD_BOOL_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// INTERMEDIATE-NEXT: [[GUARD_BOOL_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[GUARD_BOOL_TYPE]], [[GUARD_BOOL_KIND]] : i32
// INTERMEDIATE-NEXT: [[ALL_GUARDS_VALID:%[a-zA-Z0-9_]+]] = arith.andi [[GUARD_AFTER_INT]], [[GUARD_BOOL_VALID]] : i1
// INTERMEDIATE-NEXT: scf.if [[ALL_GUARDS_VALID]] {
// INTERMEDIATE: [[GUARDED_RESULT:%[a-zA-Z0-9_]+]] = func.call @guarded([[GUARD_INT]], [[GUARD_BOOL]]) : (!torch.int, !torch.bool) -> !torch.int
// INTERMEDIATE-NEXT: [[GUARDED_ABI_RESULT:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[GUARDED_RESULT]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: llvm.store [[GUARDED_ABI_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// INTERMEDIATE: [[GUARD_EXCEPTION_DEC_REF:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIObjectDecRef([[GUARD_EXCEPTION_HANDLE:%[a-zA-Z0-9_]+]]) : (!llvm.ptr) -> i32
// INTERMEDIATE-NEXT: [[GUARD_FAILURE_RESULT:%[a-zA-Z0-9_]+]] = llvm.load [[GUARD_FAILURE_RESULT_SLOT:%[a-zA-Z0-9_]+]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// INTERMEDIATE-NEXT: llvm.store [[GUARD_FAILURE_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// INTERMEDIATE-NEXT: }
// INTERMEDIATE-NEXT: [[GUARD_STATUS:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// INTERMEDIATE-NEXT: return [[GUARD_STATUS]] : i32
// FINAL-LABEL: llvm.func @__tvm_ffi_guarded(
// FINAL-NOT: scf.if
// FINAL: [[FINAL_BOOL_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// FINAL-NEXT: [[FINAL_INT_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// FINAL: [[FINAL_INT_ANY:%[a-zA-Z0-9_]+]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[FINAL_INT_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[FINAL_INT_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[FINAL_INT_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[FINAL_INT_TYPE]], [[FINAL_INT_KIND]] : i32
// FINAL-NEXT: [[FINAL_BOOL_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[FINAL_BOOL_ANY:%[a-zA-Z0-9_]+]] = llvm.load [[FINAL_BOOL_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[FINAL_BOOL_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[FINAL_BOOL_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: [[FINAL_BOOL_VALID:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[FINAL_BOOL_TYPE]], [[FINAL_BOOL_KIND]] : i32
// FINAL-NEXT: [[FINAL_GUARDS_VALID:%[a-zA-Z0-9_]+]] = llvm.and [[FINAL_INT_VALID]], [[FINAL_BOOL_VALID]] : i1
// FINAL-NEXT: llvm.cond_br [[FINAL_GUARDS_VALID]], [[FINAL_GUARD_SUCCESS:\^bb[0-9]+]], [[FINAL_GUARD_FAILURE:\^bb[0-9]+]]
// FINAL: [[FINAL_GUARD_SUCCESS]]:
// FINAL-NEXT: [[FINAL_GUARDED_RESULT:%[a-zA-Z0-9_]+]] = llvm.call @guarded([[FINAL_INT_ANY]], [[FINAL_BOOL_ANY]]) : (!llvm.struct<(i32, i32, i64)>, !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
// FINAL-NEXT: llvm.store [[FINAL_GUARDED_RESULT]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @guarded(%arg0: !torch.int, %arg1: !torch.bool) -> !torch.int attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %arg0 : !torch.int
}
