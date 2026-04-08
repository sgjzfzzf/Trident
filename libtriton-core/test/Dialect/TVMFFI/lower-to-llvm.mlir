// RUN: libtriton-core-opt %s -convert-to-llvm | FileCheck %s
// RUN: libtriton-core-opt %s -convert-to-llvm | mlir-opt -convert-func-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o /dev/null

// CHECK: llvm.func @TVMFFITensorFromDLPack

// CHECK-LABEL: func.func @lowering_from_int
// CHECK-SAME: (%[[FROM_INT_ARG:.*]]: i64) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_int(%i: i64) -> !tvm_ffi.any {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[FROM_INT_TYPE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[FROM_INT_ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[FROM_INT_INIT:.*]] = llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[FROM_INT_WITH_TYPE:.*]] = llvm.insertvalue %[[FROM_INT_TYPE]], %[[FROM_INT_INIT]][0]
  // CHECK: %[[FROM_INT_WITH_AUX:.*]] = llvm.insertvalue %[[FROM_INT_ZERO]], %[[FROM_INT_WITH_TYPE]][1]
  // CHECK: %[[FROM_INT_VALUE:.*]] = llvm.insertvalue %[[FROM_INT_ARG]], %[[FROM_INT_WITH_AUX]][2]
  // CHECK: return %[[FROM_INT_VALUE]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.from_int %i : i64 -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_int
// CHECK-SAME: (%[[TO_INT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> i64
func.func @lowering_to_int(%a: !tvm_ffi.any) -> i64 {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_INT_VALUE:.*]] = llvm.extractvalue %[[TO_INT_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: return %[[TO_INT_VALUE]] : i64
  %0 = tvm_ffi.to_int %a : !tvm_ffi.any -> i64
  return %0 : i64
}

// CHECK-LABEL: func.func @lowering_from_float
// CHECK-SAME: (%[[FROM_FLOAT_ARG:.*]]: f64) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_float(%f: f64) -> !tvm_ffi.any {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[FROM_FLOAT_TYPE:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[FROM_FLOAT_BITS:.*]] = llvm.bitcast %[[FROM_FLOAT_ARG]] : f64 to i64
  // CHECK: %[[FROM_FLOAT_ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[FROM_FLOAT_INIT:.*]] = llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[FROM_FLOAT_WITH_TYPE:.*]] = llvm.insertvalue %[[FROM_FLOAT_TYPE]], %[[FROM_FLOAT_INIT]][0]
  // CHECK: %[[FROM_FLOAT_WITH_AUX:.*]] = llvm.insertvalue %[[FROM_FLOAT_ZERO]], %[[FROM_FLOAT_WITH_TYPE]][1]
  // CHECK: %[[FROM_FLOAT_VALUE:.*]] = llvm.insertvalue %[[FROM_FLOAT_BITS]], %[[FROM_FLOAT_WITH_AUX]][2]
  // CHECK: return %[[FROM_FLOAT_VALUE]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.from_float %f : f64 -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_float
// CHECK-SAME: (%[[TO_FLOAT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> f64
func.func @lowering_to_float(%a: !tvm_ffi.any) -> f64 {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_FLOAT_BITS:.*]] = llvm.extractvalue %[[TO_FLOAT_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[TO_FLOAT_VALUE:.*]] = llvm.bitcast %[[TO_FLOAT_BITS]] : i64 to f64
  // CHECK: return %[[TO_FLOAT_VALUE]] : f64
  %0 = tvm_ffi.to_float %a : !tvm_ffi.any -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @lowering_from_str
// CHECK-SAME: (%[[FROM_STR_ARG:.*]]: !llvm.ptr) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_str(%p: !llvm.ptr) -> !tvm_ffi.any {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[FROM_STR_TYPE:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: %[[FROM_STR_BITS:.*]] = llvm.ptrtoint %[[FROM_STR_ARG]] : !llvm.ptr to i64
  // CHECK: %[[FROM_STR_ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[FROM_STR_INIT:.*]] = llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[FROM_STR_WITH_TYPE:.*]] = llvm.insertvalue %[[FROM_STR_TYPE]], %[[FROM_STR_INIT]][0]
  // CHECK: %[[FROM_STR_WITH_AUX:.*]] = llvm.insertvalue %[[FROM_STR_ZERO]], %[[FROM_STR_WITH_TYPE]][1]
  // CHECK: %[[FROM_STR_VALUE:.*]] = llvm.insertvalue %[[FROM_STR_BITS]], %[[FROM_STR_WITH_AUX]][2]
  // CHECK: return %[[FROM_STR_VALUE]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.from_str %p : !llvm.ptr -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_str
// CHECK-SAME: (%[[TO_STR_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr
func.func @lowering_to_str(%a: !tvm_ffi.any) -> !llvm.ptr {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_STR_BITS:.*]] = llvm.extractvalue %[[TO_STR_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[TO_STR_VALUE:.*]] = llvm.inttoptr %[[TO_STR_BITS]] : i64 to !llvm.ptr
  // CHECK: return %[[TO_STR_VALUE]] : !llvm.ptr
  %0 = tvm_ffi.to_str %a : !tvm_ffi.any -> !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: func.func @lowering_from_tensor
// CHECK-SAME: (%[[FROM_DLPACK_ARG:.*]]: !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>, %[[FROM_TENSOR_ALIGN:.*]]: i32, %[[FROM_TENSOR_CONTIG:.*]]: i32)
// CHECK-SAME: -> !llvm.struct<(i32, i32, i64)>
// NO-CAST-LABEL: func.func @lowering_from_tensor
func.func @lowering_from_tensor(%from: !dlpack.managed_tensor, %align: i32, %contig: i32) -> !tvm_ffi.any {
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // NO-CAST-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[FROM_SIZE:.*]] = llvm.mlir.constant(64 : i64)
  // CHECK: %[[FROM_SLOT:.*]] = llvm.call @malloc(%[[FROM_SIZE]]) : (i64) -> !llvm.ptr
  // CHECK: llvm.store %[[FROM_DLPACK_ARG]], %[[FROM_SLOT]] : !llvm.struct<(struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>, !llvm.ptr
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
  %h = tvm_ffi.tensor_from_dlpack %from, %align, %contig : !dlpack.managed_tensor, i32, i32 -> !tvm_ffi.object_handle
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
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[FROM_OBJECT_TYPE:.*]] = llvm.load %[[FROM_OBJECT_ARG]] : !llvm.ptr -> i32
  // CHECK: %[[FROM_OBJECT_BITS:.*]] = llvm.ptrtoint %[[FROM_OBJECT_ARG]] : !llvm.ptr to i64
  // CHECK: %[[FROM_OBJECT_ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[FROM_OBJECT_INIT:.*]] = llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[FROM_OBJECT_WITH_TYPE:.*]] = llvm.insertvalue %[[FROM_OBJECT_TYPE]], %[[FROM_OBJECT_INIT]][0]
  // CHECK: %[[FROM_OBJECT_WITH_AUX:.*]] = llvm.insertvalue %[[FROM_OBJECT_ZERO]], %[[FROM_OBJECT_WITH_TYPE]][1]
  // CHECK: %[[FROM_OBJECT_VALUE:.*]] = llvm.insertvalue %[[FROM_OBJECT_BITS]], %[[FROM_OBJECT_WITH_AUX]][2]
  // CHECK: return %[[FROM_OBJECT_VALUE]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.from_object %h : !tvm_ffi.object_handle -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_object
// CHECK-SAME: (%[[TO_OBJECT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr
func.func @lowering_to_object(%a: !tvm_ffi.any) -> !tvm_ffi.object_handle {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_OBJECT_BITS:.*]] = llvm.extractvalue %[[TO_OBJECT_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[TO_OBJECT_VALUE:.*]] = llvm.inttoptr %[[TO_OBJECT_BITS]] : i64 to !llvm.ptr
  // CHECK: return %[[TO_OBJECT_VALUE]] : !llvm.ptr
  %0 = tvm_ffi.to_object %a : !tvm_ffi.any -> !tvm_ffi.object_handle
  return %0 : !tvm_ffi.object_handle
}

