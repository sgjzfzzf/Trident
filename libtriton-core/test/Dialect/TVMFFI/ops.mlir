// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @any_from_i64
func.func @any_from_i64(%arg0: i64) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : i64 -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : i64 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @i64_from_any
func.func @i64_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> i64
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> i64
  return
}

// -----

// CHECK-LABEL: func.func @any_from_i32
func.func @any_from_i32(%arg0: i32) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : i32 -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : i32 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @i32_from_any
func.func @i32_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> i32
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> i32
  return
}

// -----

// CHECK-LABEL: func.func @type_index_from_any
func.func @type_index_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.get_type_index %arg0 : !tvm_ffi.any -> i32
  %0 = tvm_ffi.get_type_index %arg0 : !tvm_ffi.any -> i32
  return
}

// -----

// CHECK-LABEL: func.func @any_from_f64
func.func @any_from_f64(%arg0: f64) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : f64 -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : f64 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @i64_from_any
func.func @i64_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> i64
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> i64
  return
}

// -----

// CHECK-LABEL: func.func @any_from_i32
func.func @any_from_i32(%arg0: i32) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : i32 -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : i32 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @i32_from_any
func.func @i32_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> i32
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> i32
  return
}

// -----

// CHECK-LABEL: func.func @any_from_ptr
func.func @any_from_ptr(%arg0: !llvm.ptr) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.load %arg0 : !llvm.ptr -> !tvm_ffi.any
  %0 = tvm_ffi.load %arg0 : !llvm.ptr -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @any_from_object_handle
func.func @any_from_object_handle(%arg0: !tvm_ffi.object_handle) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.object_handle -> !tvm_ffi.any
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.object_handle -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @tensor_from_any
func.func @tensor_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> !dlpack.tensor
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> !dlpack.tensor
  return
}

// -----

// CHECK-LABEL: func.func @object_handle_from_any
func.func @object_handle_from_any(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.any -> !tvm_ffi.object_handle
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.any -> !tvm_ffi.object_handle
  return
}

// -----

// CHECK-LABEL: func.func @tensor_from_object_handle
func.func @tensor_from_object_handle(%arg0: !tvm_ffi.object_handle) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to %arg0 : !tvm_ffi.object_handle -> !dlpack.tensor
  %0 = tvm_ffi.to %arg0 : !tvm_ffi.object_handle -> !dlpack.tensor
  return
}

// CHECK-LABEL: func.func @object_handle_env_tensor_alloc
func.func @object_handle_env_tensor_alloc() {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.env_tensor_alloc dtype = f32, shape = [16, 32] : !tvm_ffi.object_handle
  %0 = tvm_ffi.env_tensor_alloc dtype = f32, shape = [16, 32] : !tvm_ffi.object_handle
  return
}

// -----

// CHECK-LABEL: func.func @object_handle_inc_ref
func.func @object_handle_inc_ref(%arg0: !tvm_ffi.object_handle) {
  // CHECK: tvm_ffi.object_inc_ref %arg0 : !tvm_ffi.object_handle
  tvm_ffi.object_inc_ref %arg0 : !tvm_ffi.object_handle
  return
}

// -----

// CHECK-LABEL: func.func @object_handle_dec_ref
func.func @object_handle_dec_ref(%arg0: !tvm_ffi.object_handle) {
  // CHECK: tvm_ffi.object_dec_ref %arg0 : !tvm_ffi.object_handle
  tvm_ffi.object_dec_ref %arg0 : !tvm_ffi.object_handle
  return
}

// -----

// CHECK-LABEL: func.func @get_opaque_ptr
func.func @get_opaque_ptr(%arg0: !tvm_ffi.object_handle) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.get_opaque_ptr %arg0 : !tvm_ffi.object_handle -> !llvm.ptr
  %0 = tvm_ffi.get_opaque_ptr %arg0 : !tvm_ffi.object_handle -> !llvm.ptr
  return
}

// -----

// CHECK-LABEL: func.func @load_tensor_from_opaque
func.func @load_tensor_from_opaque(%arg0: !llvm.ptr) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.load %arg0 : !llvm.ptr -> !dlpack.tensor
  %0 = tvm_ffi.load %arg0 : !llvm.ptr -> !dlpack.tensor
  return
}

// -----

// CHECK-LABEL: func.func @function_get_global
func.func @function_get_global() {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.function_get_global "foo" : !tvm_ffi.object_handle
  %0 = tvm_ffi.function_get_global "foo" : !tvm_ffi.object_handle
  return
}

// -----

// CHECK-LABEL: func.func @function_call_with_args
func.func @function_call_with_args(%arg0: !tvm_ffi.object_handle, %arg1: !tvm_ffi.any, %arg2: !tvm_ffi.any) -> !tvm_ffi.any {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.function_call %arg0(%arg1, %arg2) : (!tvm_ffi.any, !tvm_ffi.any) -> !tvm_ffi.any
  %0 = tvm_ffi.function_call %arg0(%arg1, %arg2) : (!tvm_ffi.any, !tvm_ffi.any) -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}
