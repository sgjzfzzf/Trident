// RUN: libtriton-core-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @from_memref
func.func @from_memref(%arg0: memref<?xf32>) {
  // CHECK: %[[VALUE:.*]] = dlpack.from_memref %arg0 : memref<?xf32> -> !dlpack.managed_tensor
  %0 = dlpack.from_memref %arg0 : memref<?xf32> -> !dlpack.managed_tensor
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