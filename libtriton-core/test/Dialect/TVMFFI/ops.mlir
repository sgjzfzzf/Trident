// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @from_int
func.func @from_int(%arg0: i64) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.from_int %arg0 : i64 -> !tvm_ffi.any
  %0 = tvm_ffi.from_int %arg0 : i64 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_int
func.func @to_int(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to_int %arg0 : !tvm_ffi.any -> i64
  %0 = tvm_ffi.to_int %arg0 : !tvm_ffi.any -> i64
  return
}

// -----

// CHECK-LABEL: func.func @from_float
func.func @from_float(%arg0: f64) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.from_float %arg0 : f64 -> !tvm_ffi.any
  %0 = tvm_ffi.from_float %arg0 : f64 -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_float
func.func @to_float(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to_float %arg0 : !tvm_ffi.any -> f64
  %0 = tvm_ffi.to_float %arg0 : !tvm_ffi.any -> f64
  return
}

// -----

// CHECK-LABEL: func.func @from_str
func.func @from_str(%arg0: !llvm.ptr) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.from_str %arg0 : !llvm.ptr -> !tvm_ffi.any
  %0 = tvm_ffi.from_str %arg0 : !llvm.ptr -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_str
func.func @to_str(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to_str %arg0 : !tvm_ffi.any -> !llvm.ptr
  %0 = tvm_ffi.to_str %arg0 : !tvm_ffi.any -> !llvm.ptr
  return
}

// -----

// CHECK-LABEL: func.func @from_tensor
func.func @from_tensor(%arg0: !dlpack.tensor) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.from_tensor %arg0 : !dlpack.tensor -> !tvm_ffi.any
  %0 = tvm_ffi.from_tensor %arg0 : !dlpack.tensor -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_tensor
func.func @to_tensor(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to_tensor %arg0 : !tvm_ffi.any -> !dlpack.tensor
  %0 = tvm_ffi.to_tensor %arg0 : !tvm_ffi.any -> !dlpack.tensor
  return
}

// -----

// CHECK-LABEL: func.func @from_object
func.func @from_object(%arg0: !llvm.ptr) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.from_object %arg0 : !llvm.ptr -> !tvm_ffi.any
  %0 = tvm_ffi.from_object %arg0 : !llvm.ptr -> !tvm_ffi.any
  return
}

// -----

// CHECK-LABEL: func.func @to_object
func.func @to_object(%arg0: !tvm_ffi.any) {
  // CHECK: %[[VALUE:.*]] = tvm_ffi.to_object %arg0 : !tvm_ffi.any -> !llvm.ptr
  %0 = tvm_ffi.to_object %arg0 : !tvm_ffi.any -> !llvm.ptr
  return
}
