// RUN: libtriton-core-opt %s -convert-tvm-ffi-to-llvm | FileCheck %s

// NOTE: The conversion pass body is not yet implemented (TODO stub).
// All CHECK patterns below verify that ops are preserved unchanged (pass-through).
// Replace these CHECK lines with expected LLVM-dialect forms once lowering is
// implemented.

// CHECK-LABEL: func.func @lowering_scaffold
func.func @lowering_scaffold(
    %i: i64,
    %f: f64,
    %p: !llvm.ptr,
  %t: !dlpack.tensor,
    %a: !tvm_ffi.any) {
  // TODO: Replace with expected LLVM-dialect form after implementing lowering.

  // CHECK: tvm_ffi.from_int
  // CHECK: tvm_ffi.to_int
  %from_int = tvm_ffi.from_int %i : i64 -> !tvm_ffi.any
  %to_int = tvm_ffi.to_int %a : !tvm_ffi.any -> i64

  // CHECK: tvm_ffi.from_float
  // CHECK: tvm_ffi.to_float
  %from_float = tvm_ffi.from_float %f : f64 -> !tvm_ffi.any
  %to_float = tvm_ffi.to_float %a : !tvm_ffi.any -> f64

  // CHECK: tvm_ffi.from_str
  // CHECK: tvm_ffi.to_str
  %from_str = tvm_ffi.from_str %p : !llvm.ptr -> !tvm_ffi.any
  %to_str = tvm_ffi.to_str %a : !tvm_ffi.any -> !llvm.ptr

  // CHECK: tvm_ffi.from_tensor
  // CHECK: tvm_ffi.to_tensor
  %from_tensor = tvm_ffi.from_tensor %t : !dlpack.tensor -> !tvm_ffi.any
  %to_tensor = tvm_ffi.to_tensor %a : !tvm_ffi.any -> !dlpack.tensor

  // CHECK: tvm_ffi.from_object
  // CHECK: tvm_ffi.to_object
  %from_object = tvm_ffi.from_object %p : !llvm.ptr -> !tvm_ffi.any
  %to_object = tvm_ffi.to_object %a : !tvm_ffi.any -> !llvm.ptr

  return
}
