// RUN: libtriton-core-opt %s -convert-func-signature-to-dlpack | FileCheck %s --check-prefix=CHECK-DLPACK

// CHECK-DLPACK-LABEL: func.func @plain(
// CHECK-DLPACK-SAME: %[[PX:.*]]: !dlpack.tensor) -> !dlpack.tensor {
// CHECK-DLPACK: return %[[PX]] : !dlpack.tensor
func.func @plain(%x: memref<f32>) -> memref<f32> {
  return %x : memref<f32>
}

// CHECK-DLPACK-LABEL: func.func @id_memref(
// CHECK-DLPACK-SAME: %[[M:.*]]: !dlpack.tensor) -> !dlpack.tensor
// CHECK-DLPACK: return %[[M]] : !dlpack.tensor
func.func @id_memref(%m: memref<?xf32>) -> memref<?xf32> {
  return %m : memref<?xf32>
}

// CHECK-DLPACK-LABEL: func.func @ret_tmp(
// CHECK-DLPACK-SAME: %[[TX:.*]]: !dlpack.tensor) -> !dlpack.tensor {
// CHECK-DLPACK: %[[TMP:.*]] = memref.alloca() : memref<4xf32>
// CHECK-DLPACK: %[[OWNED:.*]] = dlpack.from_memref_owned %[[TMP]] : memref<4xf32> -> !dlpack.managed_tensor
// CHECK-DLPACK: %[[VIEW:.*]] = dlpack.view %[[OWNED]] : !dlpack.managed_tensor -> !dlpack.tensor
// CHECK-DLPACK: return %[[VIEW]] : !dlpack.tensor
func.func @ret_tmp(%x: memref<4xf32>) -> memref<4xf32> {
  %tmp = memref.alloca() : memref<4xf32>
  return %tmp : memref<4xf32>
}
