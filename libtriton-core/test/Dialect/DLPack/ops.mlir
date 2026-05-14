// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @from_memref
func.func @from_memref(%arg0: memref<?xf32>) {
  // CHECK: %[[VALUE:.*]] = dlpack.from_memref_owned %arg0 : memref<?xf32> -> !dlpack.managed_tensor
  %0 = dlpack.from_memref_owned %arg0 : memref<?xf32> -> !dlpack.managed_tensor
  return
}

// -----

// CHECK-LABEL: func.func @view
func.func @view(%arg0: !dlpack.managed_tensor) {
  // CHECK: %[[VALUE:.*]] = dlpack.view %arg0 : !dlpack.managed_tensor -> !dlpack.tensor
  %0 = dlpack.view %arg0 : !dlpack.managed_tensor -> !dlpack.tensor
  return
}

// -----

// CHECK-LABEL: func.func @to_memref
func.func @to_memref(%arg0: !dlpack.tensor) {
  // CHECK: %[[VALUE:.*]] = dlpack.to_memref %arg0 : !dlpack.tensor -> memref<?xf32>
  %0 = dlpack.to_memref %arg0 : !dlpack.tensor -> memref<?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @ndim
func.func @ndim(%arg0: !dlpack.tensor) {
  // CHECK: %[[VALUE:.*]] = dlpack.ndim %arg0 : !dlpack.tensor -> i32
  %0 = dlpack.ndim %arg0 : !dlpack.tensor -> i32
  return
}

// -----

// CHECK-LABEL: func.func @shape
func.func @shape(%arg0: !dlpack.tensor, %arg1: index) {
  // CHECK: %[[VALUE:.*]] = dlpack.shape %arg0[%arg1] : !dlpack.tensor, index -> i64
  %0 = dlpack.shape %arg0[%arg1] : !dlpack.tensor, index -> i64
  return
}

// -----

// CHECK-LABEL: func.func @strides
func.func @strides(%arg0: !dlpack.tensor, %arg1: index) {
  // CHECK: %[[VALUE:.*]] = dlpack.strides %arg0[%arg1] : !dlpack.tensor, index -> i64
  %0 = dlpack.strides %arg0[%arg1] : !dlpack.tensor, index -> i64
  return
}

// -----

// CHECK-LABEL: func.func @byte_offset
func.func @byte_offset(%arg0: !dlpack.tensor) {
  // CHECK: %[[VALUE:.*]] = dlpack.byte_offset %arg0 : !dlpack.tensor -> i64
  %0 = dlpack.byte_offset %arg0 : !dlpack.tensor -> i64
  return
}