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

// CHECK-LABEL: func.func @get_type_index
func.func @get_type_index(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.get_type_index %arg0 : !tvm_ffi.any -> i32
  %0 = tvm_ffi.get_type_index %arg0 : !tvm_ffi.any -> i32
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

// CHECK-LABEL: func.func @as_any_from_llvm
func.func @as_any_from_llvm(%arg0: !llvm.struct<(i32, i32, i64)>) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.as %arg0 : !llvm.struct<(i32, i32, i64)> -> !tvm_ffi.any
  %0 = tvm_ffi.as %arg0 : !llvm.struct<(i32, i32, i64)> -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @as_llvm_from_any
func.func @as_llvm_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.as %arg0 : !tvm_ffi.any -> !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.as %arg0 : !tvm_ffi.any -> !llvm.struct<(i32, i32, i64)>
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

// -----

// CHECK-LABEL: func.func @env_tensor_alloc
func.func @env_tensor_alloc() {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.env_tensor_alloc dtype = f32, shape = [16, 32] : !tvm_ffi.object_handle
  %0 = tvm_ffi.env_tensor_alloc dtype = f32, shape = [16, 32] : !tvm_ffi.object_handle
  return
}

// -----

// CHECK-LABEL: func.func @object_inc_ref
func.func @object_inc_ref(%arg0: !tvm_ffi.object_handle) {
  // CHECK: tvm_ffi.object_inc_ref %arg0 : !tvm_ffi.object_handle
  tvm_ffi.object_inc_ref %arg0 : !tvm_ffi.object_handle
  return
}

// -----

// CHECK-LABEL: func.func @object_dec_ref
func.func @object_dec_ref(%arg0: !tvm_ffi.object_handle) {
  // CHECK: tvm_ffi.object_dec_ref %arg0 : !tvm_ffi.object_handle
  tvm_ffi.object_dec_ref %arg0 : !tvm_ffi.object_handle
  return
}
