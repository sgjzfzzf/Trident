// RUN: libtriton-core-opt %s -convert-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @lower_from_memref
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
// CHECK-SAME: -> !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>
func.func @lower_from_memref(%arg0: memref<?xf32>) -> !dlpack.managed_tensor {
  // CHECK: %[[ALIGNED_PTR:.*]] = llvm.extractvalue %[[ARG]][1]
  // CHECK: %[[OFFSET:.*]] = llvm.extractvalue %[[ARG]][2]
  // CHECK: %[[SHAPE_SLOT:.*]] = llvm.alloca {{.*}} x i64
  // CHECK: %[[DIM0_SIZE:.*]] = llvm.extractvalue %[[ARG]][3, 0]
  // CHECK: %[[SHAPE_GEP:.*]] = llvm.getelementptr %[[SHAPE_SLOT]][0]
  // CHECK: llvm.store %[[DIM0_SIZE]], %[[SHAPE_GEP]]
  // CHECK: %[[STRIDE_SLOT:.*]] = llvm.alloca {{.*}} x i64
  // CHECK: %[[DIM0_STRIDE:.*]] = llvm.extractvalue %[[ARG]][4, 0]
  // CHECK: %[[STRIDE_GEP:.*]] = llvm.getelementptr %[[STRIDE_SLOT]][0]
  // CHECK: llvm.store %[[DIM0_STRIDE]], %[[STRIDE_GEP]]
  // CHECK: llvm.mlir.constant(1 : i32)
  // CHECK: llvm.mlir.constant(2 : i8)
  // CHECK: llvm.mlir.constant(32 : i8)
  // CHECK: llvm.mlir.constant(1 : i16)
  // CHECK: llvm.mlir.zero : !llvm.ptr
  // CHECK: return %[[MANAGED:.*]] : !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>
  %0 = dlpack.from_memref %arg0 : memref<?xf32> -> !dlpack.managed_tensor
  return %0 : !dlpack.managed_tensor
}

// -----

// CHECK-LABEL: func.func @lower_view
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>)
// CHECK-SAME: -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
func.func @lower_view(%arg0: !dlpack.managed_tensor) -> !dlpack.tensor {
  // CHECK: %[[TENSOR_VIEW:.*]] = llvm.extractvalue %[[ARG]][0]
  // CHECK: return %[[TENSOR_VIEW]] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
  %0 = dlpack.view %arg0 : !dlpack.managed_tensor -> !dlpack.tensor
  return %0 : !dlpack.tensor
}

// -----

// CHECK-LABEL: func.func @lower_to_memref
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
func.func @lower_to_memref(%arg0: !dlpack.tensor) -> memref<?xf32> {
  // CHECK: %[[DATA_PTR:.*]] = llvm.extractvalue %[[ARG]][0]
  // CHECK: %[[SHAPE_PTR:.*]] = llvm.extractvalue %[[ARG]][4]
  // CHECK: %[[STRIDE_PTR:.*]] = llvm.extractvalue %[[ARG]][5]
  // CHECK: %[[BYTE_OFFSET:.*]] = llvm.extractvalue %[[ARG]][6]
  // CHECK: %[[SHAPE_GEP:.*]] = llvm.getelementptr %[[SHAPE_PTR]]
  // CHECK: %[[SHAPE0:.*]] = llvm.load %[[SHAPE_GEP]]
  // CHECK: %[[STRIDE_GEP:.*]] = llvm.getelementptr %[[STRIDE_PTR]]
  // CHECK: %[[STRIDE0:.*]] = llvm.load %[[STRIDE_GEP]]
  // CHECK: %[[MEMREF_WITH_SHAPE:.*]] = llvm.insertvalue %[[SHAPE0]], %{{.*}}[3, 0]
  // CHECK: %[[MEMREF_RESULT:.*]] = llvm.insertvalue %[[STRIDE0]], %[[MEMREF_WITH_SHAPE]][4, 0]
  // CHECK: return %[[MEMREF_RESULT]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %0 = dlpack.to_memref %arg0 : !dlpack.tensor -> memref<?xf32>
  return %0 : memref<?xf32>
}