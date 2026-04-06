// RUN: libtriton-core-opt %s -convert-to-llvm | FileCheck %s

// CHECK: llvm.func @TVMFFITensorFromDLPack

// CHECK-LABEL: func.func @lowering_from_int
// CHECK-SAME: (%[[FROM_INT_ARG:.*]]: i64) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_int(%i: i64) -> !tvm_ffi.any {
  // CHECK: tvm_ffi.from_int
  %0 = tvm_ffi.from_int %i : i64 -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_int
// CHECK-SAME: (%[[TO_INT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> i64
func.func @lowering_to_int(%a: !tvm_ffi.any) -> i64 {
  // CHECK: tvm_ffi.to_int
  %0 = tvm_ffi.to_int %a : !tvm_ffi.any -> i64
  return %0 : i64
}

// CHECK-LABEL: func.func @lowering_from_float
// CHECK-SAME: (%[[FROM_FLOAT_ARG:.*]]: f64) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_float(%f: f64) -> !tvm_ffi.any {
  // CHECK: tvm_ffi.from_float
  %0 = tvm_ffi.from_float %f : f64 -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_float
// CHECK-SAME: (%[[TO_FLOAT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> f64
func.func @lowering_to_float(%a: !tvm_ffi.any) -> f64 {
  // CHECK: tvm_ffi.to_float
  %0 = tvm_ffi.to_float %a : !tvm_ffi.any -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @lowering_from_str
// CHECK-SAME: (%[[FROM_STR_ARG:.*]]: !llvm.ptr) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_str(%p: !llvm.ptr) -> !tvm_ffi.any {
  // CHECK: tvm_ffi.from_str
  %0 = tvm_ffi.from_str %p : !llvm.ptr -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_str
// CHECK-SAME: (%[[TO_STR_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr
func.func @lowering_to_str(%a: !tvm_ffi.any) -> !llvm.ptr {
  // CHECK: tvm_ffi.to_str
  %0 = tvm_ffi.to_str %a : !tvm_ffi.any -> !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: func.func @lowering_from_tensor
// CHECK-SAME: (%[[FROM_DLPACK_ARG:.*]]: !llvm.ptr, %[[FROM_TENSOR_ALIGN:.*]]: i32, %[[FROM_TENSOR_CONTIG:.*]]: i32)
// CHECK-SAME: -> !llvm.struct<(i32, i32, i64)>
// NO-CAST-LABEL: func.func @lowering_from_tensor
func.func @lowering_from_tensor(%from: !llvm.ptr, %align: i32, %contig: i32) -> !tvm_ffi.any {
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // NO-CAST-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[FROM_SLOT:.*]] = llvm.alloca %[[ONE]] x i64 : (i64) -> !llvm.ptr
  // CHECK: llvm.store %[[FROM_DLPACK_ARG]], %[[FROM_SLOT]]
  // CHECK: %[[OUT_SLOT:.*]] = llvm.alloca %[[ONE]] x i64 : (i64) -> !llvm.ptr
  // CHECK: %[[ZERO_PTR:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: llvm.store %[[ZERO_PTR]], %[[OUT_SLOT]]
  // CHECK: llvm.call @TVMFFITensorFromDLPack(%[[FROM_SLOT]],
  // CHECK-SAME: %[[FROM_TENSOR_ALIGN]], %[[FROM_TENSOR_CONTIG]], %[[OUT_SLOT]])
  // CHECK: %[[HANDLE:.*]] = llvm.load %[[OUT_SLOT]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[PAYLOAD_BITS:.*]] = llvm.ptrtoint %[[HANDLE]] : !llvm.ptr to i64
  // CHECK: %[[ANY_INIT:.*]] = llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[ANY_WITH_TYPE:.*]] = llvm.insertvalue %{{.*}}, %[[ANY_INIT]][0]
  // CHECK: %[[ANY_WITH_AUX:.*]] = llvm.insertvalue %{{.*}}, %[[ANY_WITH_TYPE]][1]
  // CHECK: %[[ANY_VALUE:.*]] = llvm.insertvalue %[[PAYLOAD_BITS]], %[[ANY_WITH_AUX]][2]
  // CHECK-NOT: tvm_ffi.tensor_from_dlpack
  // CHECK-NOT: tvm_ffi.from_tensor
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[ANY_VALUE]] : !llvm.struct<(i32, i32, i64)>
  // NO-CAST: return
  %h = tvm_ffi.tensor_from_dlpack %from, %align, %contig : !llvm.ptr, i32, i32 -> !tvm_ffi.object_handle
  %0 = tvm_ffi.from_tensor %h : !tvm_ffi.object_handle -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_tensor
// CHECK-SAME: (%[[TO_TENSOR_ARG:.*]]: !llvm.struct<(i32, i32, i64)>)
// CHECK-SAME: -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// NO-CAST-LABEL: func.func @lowering_to_tensor
func.func @lowering_to_tensor(%a: !tvm_ffi.any) -> !dlpack.tensor {
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // NO-CAST-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[PAYLOAD_BITS:.*]] = llvm.extractvalue %[[TO_TENSOR_ARG]][2]
  // CHECK: %[[PAYLOAD_PTR:.*]] = llvm.inttoptr %[[PAYLOAD_BITS]] : i64 to !llvm.ptr
  // CHECK: %[[TENSOR_CELL_PTR:.*]] = llvm.getelementptr %[[PAYLOAD_PTR]][24]
  // CHECK: %[[TENSOR_VALUE:.*]] = llvm.load %[[TENSOR_CELL_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
  // CHECK-NOT: tvm_ffi.to_tensor
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[TENSOR_VALUE]] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
  // NO-CAST: return
  %0 = tvm_ffi.to_tensor %a : !tvm_ffi.any -> !dlpack.tensor
  return %0 : !dlpack.tensor
}

// CHECK-LABEL: func.func @lowering_from_object
// CHECK-SAME: (%[[FROM_OBJECT_ARG:.*]]: !llvm.ptr) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_object(%h: !tvm_ffi.object_handle) -> !tvm_ffi.any {
  // CHECK: tvm_ffi.from_object
  %0 = tvm_ffi.from_object %h : !tvm_ffi.object_handle -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_object
// CHECK-SAME: (%[[TO_OBJECT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr
func.func @lowering_to_object(%a: !tvm_ffi.any) -> !tvm_ffi.object_handle {
  // CHECK: tvm_ffi.to_object
  %0 = tvm_ffi.to_object %a : !tvm_ffi.any -> !tvm_ffi.object_handle
  return %0 : !tvm_ffi.object_handle
}

