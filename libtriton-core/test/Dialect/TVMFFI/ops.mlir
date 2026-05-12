// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @to_any_from_int
func.func @to_any_from_int(%arg0: i64) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : i64 -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : i64 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_int_from_any
func.func @to_int_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> i64
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> i64
  return
}

// -----

// CHECK-LABEL: func.func @to_any_from_i32
func.func @to_any_from_i32(%arg0: i32) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : i32 -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : i32 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_i32_from_any
func.func @to_i32_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> i32
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> i32
  return
}

// -----

// CHECK-LABEL: func.func @to_any_from_float
func.func @to_any_from_float(%arg0: f64) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : f64 -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : f64 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_float_from_any
func.func @to_float_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> f64
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> f64
  return
}

// -----

// CHECK-LABEL: func.func @to_any_from_str
func.func @to_any_from_str(%arg0: !llvm.ptr) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !llvm.ptr -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : !llvm.ptr -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_str_from_any
func.func @to_str_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> !llvm.ptr
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> !llvm.ptr
  return
}

// -----

// CHECK-LABEL: func.func @to_any_from_object
func.func @to_any_from_object(%arg0: !tvm_ffi.object_handle) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.object_handle -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.object_handle -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_tensor_from_any
func.func @to_tensor_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> !dlpack.tensor
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> !dlpack.tensor
  return
}

// -----

// CHECK-LABEL: func.func @to_object_from_any
func.func @to_object_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> !tvm_ffi.object_handle
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> !tvm_ffi.object_handle
  return
}

// -----

// CHECK-LABEL: func.func @tensor_from_dlpack
func.func @tensor_from_dlpack(%arg0: !dlpack.managed_tensor, %arg1: i32, %arg2: i32) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.tensor_from_dlpack %arg0, %arg1, %arg2 : !dlpack.managed_tensor, i32, i32 -> !tvm_ffi.object_handle
  %0 = tvm_ffi.tensor_from_dlpack %arg0, %arg1, %arg2 : !dlpack.managed_tensor, i32, i32 -> !tvm_ffi.object_handle
  return
}
