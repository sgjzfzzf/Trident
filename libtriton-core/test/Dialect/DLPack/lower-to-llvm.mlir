// RUN: libtriton-core-opt %s -convert-to-llvm | FileCheck %s
// RUN: libtriton-core-opt %s -convert-to-llvm | mlir-opt -convert-func-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o /dev/null

// CHECK-LABEL: func.func @lower_view_managed_tensor
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>)
// CHECK-SAME: -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
func.func @lower_view_managed_tensor(%arg0: !dlpack.managed_tensor) -> !dlpack.tensor {
  // CHECK: %[[TENSOR_VIEW:.*]] = llvm.extractvalue %[[ARG]][0]
  // CHECK: return %[[TENSOR_VIEW]] : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  %0 = dlpack.view %arg0 : !dlpack.managed_tensor -> !dlpack.tensor
  return %0 : !dlpack.tensor
}

// -----

// CHECK-LABEL: func.func @lower_to_memref_tensor
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
func.func @lower_to_memref_tensor(%arg0: !dlpack.tensor) -> memref<?xf32> {
  // CHECK: %[[DATA_PTR:.*]] = llvm.extractvalue %[[ARG]][0]
  // CHECK: %[[SHAPE_PTR:.*]] = llvm.extractvalue %[[ARG]][4]
  // CHECK: %[[STRIDE_PTR:.*]] = llvm.extractvalue %[[ARG]][5]
  // CHECK: %[[BYTE_OFFSET:.*]] = llvm.extractvalue %[[ARG]][6]
  // CHECK: llvm.insertvalue %[[DATA_PTR]], %{{.*}}[0]
  // CHECK: llvm.insertvalue %[[DATA_PTR]], %{{.*}}[1]
  // CHECK: llvm.insertvalue %[[BYTE_OFFSET]], %{{.*}}[2]
  // CHECK: %[[SHAPE_GEP:.*]] = llvm.getelementptr %[[SHAPE_PTR]]
  // CHECK: %[[SHAPE0:.*]] = llvm.load %[[SHAPE_GEP]]
  // CHECK: %[[NULL_PTR:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[IS_NULL_STRIDE:.*]] = llvm.icmp "eq" %[[STRIDE_PTR]], %[[NULL_PTR]] : !llvm.ptr
  // CHECK: %[[EFFECTIVE_STRIDE_PTR:.*]] = llvm.select %[[IS_NULL_STRIDE]], %{{.*}}, %[[STRIDE_PTR]] : i1, !llvm.ptr
  // CHECK: %[[STRIDE_GEP:.*]] = llvm.getelementptr %[[EFFECTIVE_STRIDE_PTR]]
  // CHECK: %[[STRIDE0:.*]] = llvm.load %[[STRIDE_GEP]]
  // CHECK: %[[MEMREF_WITH_SHAPE:.*]] = llvm.insertvalue %[[SHAPE0]], %{{.*}}[3, 0]
  // CHECK: %[[MEMREF_RESULT:.*]] = llvm.insertvalue %[[STRIDE0]], %[[MEMREF_WITH_SHAPE]][4, 0]
  // CHECK: return %[[MEMREF_RESULT]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %0 = dlpack.to_memref %arg0 : !dlpack.tensor -> memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

// CHECK-LABEL: func.func @lower_tensor_ndim
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> i32
func.func @lower_tensor_ndim(%arg0: !dlpack.tensor) -> i32 {
  // CHECK: %[[NDIM:.*]] = llvm.extractvalue %[[ARG]][2]
  // CHECK: return %[[NDIM]] : i32
  %0 = dlpack.ndim %arg0 : !dlpack.tensor -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func @lower_tensor_shape
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>, %[[IDX:.*]]: i64)
// CHECK-SAME: -> i64
func.func @lower_tensor_shape(%arg0: !dlpack.tensor, %arg1: index) -> i64 {
  // CHECK: %[[SHAPE_PTR:.*]] = llvm.extractvalue %[[ARG]][4]
  // CHECK: %[[SHAPE_GEP:.*]] = llvm.getelementptr %[[SHAPE_PTR]][%[[IDX]]]
  // CHECK: %[[SHAPE:.*]] = llvm.load %[[SHAPE_GEP]]
  // CHECK: return %[[SHAPE]] : i64
  %0 = dlpack.shape %arg0[%arg1] : !dlpack.tensor, index -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @lower_tensor_strides
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>, %[[IDX:.*]]: i64)
// CHECK-SAME: -> i64
func.func @lower_tensor_strides(%arg0: !dlpack.tensor, %arg1: index) -> i64 {
  // CHECK: %[[STRIDES_PTR:.*]] = llvm.extractvalue %[[ARG]][5]
  // CHECK: %[[STRIDES_GEP:.*]] = llvm.getelementptr %[[STRIDES_PTR]][%[[IDX]]]
  // CHECK: %[[STRIDE:.*]] = llvm.load %[[STRIDES_GEP]]
  // CHECK: return %[[STRIDE]] : i64
  %0 = dlpack.strides %arg0[%arg1] : !dlpack.tensor, index -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @lower_tensor_byte_offset
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> i64
func.func @lower_tensor_byte_offset(%arg0: !dlpack.tensor) -> i64 {
  // CHECK: %[[OFFSET:.*]] = llvm.extractvalue %[[ARG]][6]
  // CHECK: return %[[OFFSET]] : i64
  %0 = dlpack.byte_offset %arg0 : !dlpack.tensor -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: func.func @lower_tensor_dtype
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<packed (i8, i8, i16)>
func.func @lower_tensor_dtype(%arg0: !dlpack.tensor) -> !dlpack.datatype {
  // CHECK: %[[DTYPE:.*]] = llvm.extractvalue %[[ARG]][3]
  // CHECK: return %[[DTYPE]] : !llvm.struct<packed (i8, i8, i16)>
  %0 = dlpack.dtype %arg0 : !dlpack.tensor -> !dlpack.datatype
  return %0 : !dlpack.datatype
}

// -----

// CHECK-LABEL: func.func @lower_datatype_code
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (i8, i8, i16)>) -> i8
func.func @lower_datatype_code(%arg0: !dlpack.datatype) -> i8 {
  // CHECK: %[[CODE:.*]] = llvm.extractvalue %[[ARG]][0]
  // CHECK: return %[[CODE]] : i8
  %0 = dlpack.dtype_code %arg0 : !dlpack.datatype -> i8
  return %0 : i8
}

// -----

// CHECK-LABEL: func.func @lower_datatype_bits
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (i8, i8, i16)>) -> i8
func.func @lower_datatype_bits(%arg0: !dlpack.datatype) -> i8 {
  // CHECK: %[[BITS:.*]] = llvm.extractvalue %[[ARG]][1]
  // CHECK: return %[[BITS]] : i8
  %0 = dlpack.dtype_bits %arg0 : !dlpack.datatype -> i8
  return %0 : i8
}

// -----

// CHECK-LABEL: func.func @lower_datatype_lanes
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (i8, i8, i16)>) -> i16
func.func @lower_datatype_lanes(%arg0: !dlpack.datatype) -> i16 {
  // CHECK: %[[LANES:.*]] = llvm.extractvalue %[[ARG]][2]
  // CHECK: return %[[LANES]] : i16
  %0 = dlpack.dtype_lanes %arg0 : !dlpack.datatype -> i16
  return %0 : i16
}

// -----

// CHECK-LABEL: func.func @lower_tensor_device
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<packed (i32, i32)>
func.func @lower_tensor_device(%arg0: !dlpack.tensor) -> !dlpack.device {
  // CHECK: %[[DEVICE:.*]] = llvm.extractvalue %[[ARG]][1]
  // CHECK: return %[[DEVICE]] : !llvm.struct<packed (i32, i32)>
  %0 = dlpack.device %arg0 : !dlpack.tensor -> !dlpack.device
  return %0 : !dlpack.device
}

// -----

// CHECK-LABEL: func.func @lower_device_type
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (i32, i32)>) -> i32
func.func @lower_device_type(%arg0: !dlpack.device) -> i32 {
  // CHECK: %[[TYPE:.*]] = llvm.extractvalue %[[ARG]][0]
  // CHECK: return %[[TYPE]] : i32
  %0 = dlpack.device_type %arg0 : !dlpack.device -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func @lower_device_id
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<packed (i32, i32)>) -> i32
func.func @lower_device_id(%arg0: !dlpack.device) -> i32 {
  // CHECK: %[[ID:.*]] = llvm.extractvalue %[[ARG]][1]
  // CHECK: return %[[ID]] : i32
  %0 = dlpack.device_id %arg0 : !dlpack.device -> i32
  return %0 : i32
}