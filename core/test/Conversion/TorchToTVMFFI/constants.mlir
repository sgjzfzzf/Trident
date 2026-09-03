//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// Constants are converted to their semantic TVM FFI scalar types, rather than
// being lowered through the LLVM representation used by the final ABI pass.
// CHECK-LABEL: func.func @constants(
// CHECK-SAME: -> !tvm_ffi.array {
// CHECK: %[[I0:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 7
// CHECK: %[[I1:[a-zA-Z0-9_]+]] = tvm_ffi.constant.int 9
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = "tvm_ffi.array.create"(%[[I0]], %[[I1]]) : (!tvm_ffi.int, !tvm_ffi.int) -> !tvm_ffi.array
// CHECK: return %[[ARRAY]] : !tvm_ffi.array
func.func @constants() -> !torch.list<int> {
  %i = torch.constant.int 7
  %j = torch.constant.int 9
  %array = torch.prim.ListConstruct %i, %j
      : (!torch.int, !torch.int) -> !torch.list<int>
  return %array : !torch.list<int>
}

// CHECK-LABEL: func.func @constant_float() -> !tvm_ffi.float {
// CHECK: %[[FLOAT:[a-zA-Z0-9_]+]] = tvm_ffi.constant.float 2.500000e+00
// CHECK: return %[[FLOAT]] : !tvm_ffi.float
func.func @constant_float() -> !torch.float {
  %0 = torch.constant.float 2.5
  return %0 : !torch.float
}

// CHECK-LABEL: func.func @constant_string() -> !tvm_ffi.union<!tvm_ffi.raw_str, !tvm_ffi.small_str, !tvm_ffi.str> {
// CHECK: %[[RAW_STRING:[a-zA-Z0-9_]+]] = tvm_ffi.constant.raw_str "trident"
// CHECK: %[[STRING_FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.String" : !tvm_ffi.function
// CHECK: %[[STRING:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[STRING_FUNC]](%[[RAW_STRING]]) : (!tvm_ffi.raw_str) -> !tvm_ffi.union<!tvm_ffi.raw_str, !tvm_ffi.small_str, !tvm_ffi.str>
// CHECK: return %[[STRING]] : !tvm_ffi.union<!tvm_ffi.raw_str, !tvm_ffi.small_str, !tvm_ffi.str>
func.func @constant_string() -> !torch.str {
  %0 = torch.constant.str "trident"
  return %0 : !torch.str
}

// CHECK-LABEL: func.func @constant_bool() -> !tvm_ffi.bool {
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = tvm_ffi.constant.bool true
// CHECK: return %[[BOOL]] : !tvm_ffi.bool
func.func @constant_bool() -> !torch.bool {
  %0 = torch.constant.bool true
  return %0 : !torch.bool
}

// CHECK-LABEL: func.func @constant_none() -> !tvm_ffi.none {
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = tvm_ffi.constant.none
// CHECK: return %[[NONE]] : !tvm_ffi.none
func.func @constant_none() -> !torch.none {
  %0 = torch.constant.none
  return %0 : !torch.none
}
