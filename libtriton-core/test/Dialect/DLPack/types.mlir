// RUN: libtriton-core-opt %s | FileCheck %s

// CHECK-LABEL: func.func @roundtrip_dltensor_type
// CHECK-SAME: (%[[ARG:.*]]: !dlpack.tensor)
// CHECK: return %[[ARG]] : !dlpack.tensor
func.func @roundtrip_dltensor_type(%arg0: !dlpack.tensor) -> !dlpack.tensor {
  return %arg0 : !dlpack.tensor
}

// -----

// CHECK-LABEL: func.func @roundtrip_dlmanagedtensor_type
// CHECK-SAME: (%[[ARG:.*]]: !dlpack.managed_tensor)
// CHECK: return %[[ARG]] : !dlpack.managed_tensor
func.func @roundtrip_dlmanagedtensor_type(%arg0: !dlpack.managed_tensor) -> !dlpack.managed_tensor {
  return %arg0 : !dlpack.managed_tensor
}
