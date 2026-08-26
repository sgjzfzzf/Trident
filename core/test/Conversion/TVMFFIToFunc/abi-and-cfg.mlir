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

// make_int: a single scalar result is cast to its converted (TVMFFIAny)
// type and stored directly into %arg3.
tvm_ffi.func @make_int() -> !torch.int attributes {emit_tvm_ffi_abi} {
  %0 = torch.constant.int 42
  tvm_ffi.return %0 : !torch.int
}

// -----

// multi_return_three_tensors: three object elements → three element DecRefs
// after the container store.
// * Allocate three slots and fill each with a casted tensor.
// * ffi.Array call helper + kTVMFFIArray re-tag + store into %arg3.
// The input tensors are borrowed from the caller, so only the temporary
// ffi.Array function handle is released here; no element DecRef is emitted.
tvm_ffi.func @multi_return_three_tensors(%arg0: !torch.tensor, %arg1: !torch.tensor, %arg2: !torch.tensor) -> (!torch.tensor, !torch.tensor, !torch.tensor) attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %arg0, %arg1, %arg2 : !torch.tensor, !torch.tensor, !torch.tensor
}

// -----

// TVMFFIToFunc preserves CFG block argument types and forwards values directly.
// The unified LLVM conversion later converts both the branch and block.
// INTERMEDIATE-LABEL: func.func @branch_argument(
// INTERMEDIATE: [[BRANCH_VALUE:%[a-zA-Z0-9_]+]] = torch.constant.int 7
// INTERMEDIATE-NEXT: cf.br [[BRANCH_DEST:\^bb[0-9]+]]([[BRANCH_VALUE]] : !torch.int)
// INTERMEDIATE: [[BRANCH_DEST]]([[BRANCH_BLOCK_ARG:%[a-zA-Z0-9_]+]]: !torch.int):
// INTERMEDIATE-NEXT: return [[BRANCH_BLOCK_ARG]] : !torch.int
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
tvm_ffi.func @guarded(%arg0: !torch.int, %arg1: !torch.bool) -> !torch.int attributes {emit_tvm_ffi_abi} {
  tvm_ffi.return %arg0 : !torch.int
}

