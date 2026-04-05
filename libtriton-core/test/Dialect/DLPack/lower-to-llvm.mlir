// RUN: libtriton-core-opt %s --convert-dlpack-to-llvm --convert-to-llvm | FileCheck %s

// CHECK-LABEL: func.func @lower_from_memref
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
// CHECK-SAME: -> !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>
func.func @lower_from_memref(%arg0: memref<?xf32>) -> !dlpack.mtensor {
  // CHECK: llvm.extractvalue %[[ARG]][1]
  // CHECK: llvm.extractvalue %[[ARG]][2]
  // CHECK: llvm.alloca {{.*}} x i64
  // CHECK: llvm.extractvalue %[[ARG]][3, 0]
  // CHECK: llvm.store
  // CHECK: llvm.alloca {{.*}} x i64
  // CHECK: llvm.extractvalue %[[ARG]][4, 0]
  // CHECK: llvm.store
  // CHECK: llvm.mlir.constant(1 : i32)
  // CHECK: llvm.mlir.constant(2 : i8)
  // CHECK: llvm.mlir.constant(32 : i8)
  // CHECK: llvm.mlir.constant(1 : i16)
  // CHECK: llvm.mlir.zero : !llvm.ptr
  // CHECK: return {{.*}} : !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>
  %0 = dlpack.from_memref %arg0 : memref<?xf32> -> !dlpack.mtensor
  return %0 : !dlpack.mtensor
}

// -----

// CHECK-LABEL: func.func @lower_view
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>)
// CHECK-SAME: -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
func.func @lower_view(%arg0: !dlpack.mtensor) -> !dlpack.tensor {
  // CHECK: llvm.extractvalue %[[ARG]][0]
  // CHECK: return {{.*}} : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
  %0 = dlpack.view %arg0 : !dlpack.mtensor -> !dlpack.tensor
  return %0 : !dlpack.tensor
}

// -----

// CHECK-LABEL: func.func @lower_to_memref
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
func.func @lower_to_memref(%arg0: !dlpack.tensor) -> memref<?xf32> {
  // CHECK: llvm.extractvalue %[[ARG]][0]
  // CHECK: llvm.extractvalue %[[ARG]][4]
  // CHECK: llvm.extractvalue %[[ARG]][5]
  // CHECK: llvm.extractvalue %[[ARG]][6]
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  // CHECK: llvm.insertvalue {{.*}}[3, 0]
  // CHECK: llvm.insertvalue {{.*}}[4, 0]
  // CHECK: return {{.*}} : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %0 = dlpack.to_memref %arg0 : !dlpack.tensor -> memref<?xf32>
  return %0 : memref<?xf32>
}