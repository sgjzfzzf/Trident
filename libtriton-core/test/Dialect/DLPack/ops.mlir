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

// -----

// CHECK-LABEL: func.func @dtype
func.func @dtype(%arg0: !dlpack.tensor) {
  // CHECK: %[[VALUE:.*]] = dlpack.dtype %arg0 : !dlpack.tensor -> !dlpack.datatype
  %0 = dlpack.dtype %arg0 : !dlpack.tensor -> !dlpack.datatype
  return
}

// -----

// CHECK-LABEL: func.func @dtype_code
func.func @dtype_code(%arg0: !dlpack.datatype) {
  // CHECK: %[[VALUE:.*]] = dlpack.dtype_code %arg0 : !dlpack.datatype -> i8
  %0 = dlpack.dtype_code %arg0 : !dlpack.datatype -> i8
  return
}

// -----

// CHECK-LABEL: func.func @dtype_bits
func.func @dtype_bits(%arg0: !dlpack.datatype) {
  // CHECK: %[[VALUE:.*]] = dlpack.dtype_bits %arg0 : !dlpack.datatype -> i8
  %0 = dlpack.dtype_bits %arg0 : !dlpack.datatype -> i8
  return
}

// -----

// CHECK-LABEL: func.func @dtype_lanes
func.func @dtype_lanes(%arg0: !dlpack.datatype) {
  // CHECK: %[[VALUE:.*]] = dlpack.dtype_lanes %arg0 : !dlpack.datatype -> i16
  %0 = dlpack.dtype_lanes %arg0 : !dlpack.datatype -> i16
  return
}

// -----

// CHECK-LABEL: func.func @device
func.func @device(%arg0: !dlpack.tensor) {
  // CHECK: %[[VALUE:.*]] = dlpack.device %arg0 : !dlpack.tensor -> !dlpack.device
  %0 = dlpack.device %arg0 : !dlpack.tensor -> !dlpack.device
  return
}

// -----

// CHECK-LABEL: func.func @device_type
func.func @device_type(%arg0: !dlpack.device) {
  // CHECK: %[[VALUE:.*]] = dlpack.device_type %arg0 : !dlpack.device -> i32
  %0 = dlpack.device_type %arg0 : !dlpack.device -> i32
  return
}

// -----

// CHECK-LABEL: func.func @device_id
func.func @device_id(%arg0: !dlpack.device) {
  // CHECK: %[[VALUE:.*]] = dlpack.device_id %arg0 : !dlpack.device -> i32
  %0 = dlpack.device_id %arg0 : !dlpack.device -> i32
  return
}