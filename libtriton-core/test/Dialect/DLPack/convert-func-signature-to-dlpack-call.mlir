// RUN: libtriton-core-opt %s -convert-func-signature-to-dlpack | FileCheck %s --check-prefix=CHECK-DLPACK

func.func @id_memref(%m: memref<?xf32>) -> memref<?xf32> {
  return %m : memref<?xf32>
}

// CHECK-DLPACK-LABEL: func.func @adapted(
// CHECK-DLPACK-SAME: %[[ARG0:.*]]: !dlpack.tensor, %[[ARG1:.*]]: !dlpack.tensor) -> !dlpack.tensor
// CHECK-DLPACK: %[[COUT_TENSOR:.*]] = call @id_memref(%[[ARG0]]) : (!dlpack.tensor) -> !dlpack.tensor
// CHECK-DLPACK: return %[[COUT_TENSOR]] : !dlpack.tensor
func.func @adapted(%a: memref<?xf32>, %b: memref<?xf32>) -> memref<?xf32> {
  %0 = call @id_memref(%a) : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}

// CHECK-DLPACK-LABEL: func.func @adapted_local(
// CHECK-DLPACK-SAME: %[[ARG0:.*]]: !dlpack.tensor) -> !dlpack.tensor {
// CHECK-DLPACK: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DLPACK: %[[TMP:.*]] = memref.alloca(%[[C4]]) : memref<?xf32>
// CHECK-DLPACK: %[[OWNED:.*]] = dlpack.from_memref_owned %[[TMP]] : memref<?xf32> -> !dlpack.managed_tensor
// CHECK-DLPACK: %[[VIEW:.*]] = dlpack.view %[[OWNED]] : !dlpack.managed_tensor -> !dlpack.tensor
// CHECK-DLPACK: %[[CALL:.*]] = call @id_memref(%[[VIEW]]) : (!dlpack.tensor) -> !dlpack.tensor
// CHECK-DLPACK: return %[[CALL]] : !dlpack.tensor
func.func @adapted_local(%a: memref<?xf32>) -> memref<?xf32> {
  %c4 = arith.constant 4 : index
  %tmp = memref.alloca(%c4) : memref<?xf32>
  %0 = call @id_memref(%tmp) : (memref<?xf32>) -> memref<?xf32>
  return %0 : memref<?xf32>
}
