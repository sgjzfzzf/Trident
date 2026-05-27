// RUN: libtriton-core-opt %s -convert-to-llvm | FileCheck %s
// RUN: libtriton-core-opt %s -convert-to-llvm | mlir-opt -convert-func-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o /dev/null

// CHECK-DAG: llvm.func @TVMFFIErrorSetRaisedFromCStr
// CHECK-DAG: llvm.func @TVMFFIObjectDecRef
// CHECK-DAG: llvm.func @TVMFFIObjectIncRef
// CHECK-DAG: llvm.func @TVMFFITensorFromDLPack
// CHECK-DAG: llvm.func @__libtriton_tvmffi_env_tensor_alloc

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
  %0 = tvm_ffi.to %i : i64 -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_int
// CHECK-SAME: (%[[TO_INT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> i64
func.func @lowering_to_int(%a: !tvm_ffi.any) -> i64 {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_INT_VALUE:.*]] = llvm.extractvalue %[[TO_INT_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: return %[[TO_INT_VALUE]] : i64
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> i64
  return %0 : i64
}

// CHECK-LABEL: func.func @lowering_from_i32
// CHECK-SAME: (%[[FROM_I32_ARG:.*]]: i32) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_i32(%i: i32) -> !tvm_ffi.any {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[FROM_I32_TYPE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[FROM_I32_BITS:.*]] = llvm.zext %[[FROM_I32_ARG]] : i32 to i64
  // CHECK: %[[FROM_I32_ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[FROM_I32_INIT:.*]] = llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[FROM_I32_WITH_TYPE:.*]] = llvm.insertvalue %[[FROM_I32_TYPE]], %[[FROM_I32_INIT]][0]
  // CHECK: %[[FROM_I32_WITH_AUX:.*]] = llvm.insertvalue %[[FROM_I32_ZERO]], %[[FROM_I32_WITH_TYPE]][1]
  // CHECK: %[[FROM_I32_VALUE:.*]] = llvm.insertvalue %[[FROM_I32_BITS]], %[[FROM_I32_WITH_AUX]][2]
  // CHECK: return %[[FROM_I32_VALUE]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.to %i : i32 -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_as_from_llvm
// CHECK-SAME: (%[[AS_FROM_LLVM_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_as_from_llvm(%a: !llvm.struct<(i32, i32, i64)>) -> !tvm_ffi.any {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[AS_FROM_LLVM_ARG]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.as %a : !llvm.struct<(i32, i32, i64)> -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_as_to_llvm
// CHECK-SAME: (%[[AS_TO_LLVM_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_as_to_llvm(%a: !tvm_ffi.any) -> !llvm.struct<(i32, i32, i64)> {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[AS_TO_LLVM_ARG]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.as %a : !tvm_ffi.any -> !llvm.struct<(i32, i32, i64)>
  return %0 : !llvm.struct<(i32, i32, i64)>
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
  %0 = tvm_ffi.to %f : f64 -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_get_type_index
// CHECK-SAME: (%[[GET_TYPE_INDEX_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> i32
func.func @lowering_get_type_index(%a: !tvm_ffi.any) -> i32 {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[GET_TYPE_INDEX_VALUE:.*]] = llvm.extractvalue %[[GET_TYPE_INDEX_ARG]][0] : !llvm.struct<(i32, i32, i64)>
  // CHECK: return %[[GET_TYPE_INDEX_VALUE]] : i32
  %0 = tvm_ffi.get_type_index %a : !tvm_ffi.any -> i32
  return %0 : i32
}

// CHECK-LABEL: func.func @lowering_to_float
// CHECK-SAME: (%[[TO_FLOAT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> f64
func.func @lowering_to_float(%a: !tvm_ffi.any) -> f64 {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_FLOAT_BITS:.*]] = llvm.extractvalue %[[TO_FLOAT_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[TO_FLOAT_VALUE:.*]] = llvm.bitcast %[[TO_FLOAT_BITS]] : i64 to f64
  // CHECK: return %[[TO_FLOAT_VALUE]] : f64
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> f64
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
  %0 = tvm_ffi.to %p : !llvm.ptr -> !tvm_ffi.any
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
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: func.func @lowering_error_set_raised_from_c_str
// CHECK-SAME: (%[[ERROR_KIND:.*]]: !llvm.ptr, %[[ERROR_MESSAGE:.*]]: !llvm.ptr)
func.func @lowering_error_set_raised_from_c_str(%kind: !llvm.ptr, %message: !llvm.ptr) {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: llvm.call @TVMFFIErrorSetRaisedFromCStr(%[[ERROR_KIND]], %[[ERROR_MESSAGE]]) : (!llvm.ptr, !llvm.ptr) -> ()
  // CHECK: return
  tvm_ffi.error_set_raised_from_c_str %kind, %message : !llvm.ptr, !llvm.ptr
  return
}

// CHECK-LABEL: func.func @lowering_object_inc_ref
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @lowering_object_inc_ref(%obj: !tvm_ffi.object_handle) {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: llvm.call @TVMFFIObjectIncRef(%[[OBJ]]) : (!llvm.ptr) -> i32
  // CHECK: return
  tvm_ffi.object_inc_ref %obj : !tvm_ffi.object_handle
  return
}

// CHECK-LABEL: func.func @lowering_object_dec_ref
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @lowering_object_dec_ref(%obj: !tvm_ffi.object_handle) {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: llvm.call @TVMFFIObjectDecRef(%[[OBJ]]) : (!llvm.ptr) -> i32
  // CHECK: return
  tvm_ffi.object_dec_ref %obj : !tvm_ffi.object_handle
  return
}

// CHECK-LABEL: func.func @lowering_from_tensor
// CHECK-SAME: (%[[FROM_DLPACK_ARG:.*]]: !llvm.struct<packed (struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>, %[[FROM_TENSOR_ALIGN:.*]]: i32, %[[FROM_TENSOR_CONTIG:.*]]: i32)
// CHECK-SAME: -> !llvm.struct<(i32, i32, i64)>
// NO-CAST-LABEL: func.func @lowering_from_tensor
func.func @lowering_from_tensor(%from: !dlpack.managed_tensor, %align: i32, %contig: i32) -> !tvm_ffi.any {
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // NO-CAST-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[FROM_SIZE:.*]] = llvm.mlir.constant(64 : i64)
  // CHECK: %[[FROM_SLOT:.*]] = llvm.call @malloc(%[[FROM_SIZE]]) : (i64) -> !llvm.ptr
  // CHECK: llvm.store %[[FROM_DLPACK_ARG]], %[[FROM_SLOT]] : !llvm.struct<packed (struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>, ptr, ptr)>, !llvm.ptr
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
  // CHECK-NOT: tvm_ffi.to
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[ANY_VALUE]] : !llvm.struct<(i32, i32, i64)>
  // NO-CAST: return
  %h = tvm_ffi.tensor_from_dlpack %from, %align, %contig : !dlpack.managed_tensor, i32, i32 -> !tvm_ffi.object_handle
  %0 = tvm_ffi.to %h : !tvm_ffi.object_handle -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lowering_to_tensor
// CHECK-SAME: (%[[TO_TENSOR_ARG:.*]]: !llvm.struct<(i32, i32, i64)>)
// CHECK-SAME: -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
// NO-CAST-LABEL: func.func @lowering_to_tensor
func.func @lowering_to_tensor(%a: !tvm_ffi.any) -> !dlpack.tensor {
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // NO-CAST-NOT: builtin.unrealized_conversion_cast
  // CHECK-DAG: %[[TYPE_INDEX:.*]] = llvm.extractvalue %[[TO_TENSOR_ARG]][0]
  // CHECK-DAG: %[[PAYLOAD_BITS:.*]] = llvm.extractvalue %[[TO_TENSOR_ARG]][2]
  // CHECK-DAG: %[[TENSOR_TYPE:.*]] = llvm.mlir.constant(70 : i32)
  // CHECK-DAG: %[[IS_TENSOR:.*]] = llvm.icmp "eq" %[[TYPE_INDEX]], %[[TENSOR_TYPE]] : i32
  // CHECK-DAG: %[[PAYLOAD_PTR:.*]] = llvm.inttoptr %[[PAYLOAD_BITS]] : i64 to !llvm.ptr
  // CHECK-DAG: %[[TENSOR_CELL_PTR:.*]] = llvm.getelementptr %[[PAYLOAD_PTR]][24]
  // CHECK: %[[SELECTED_TENSOR_PTR:.*]] = llvm.select %[[IS_TENSOR]], %[[TENSOR_CELL_PTR]], %[[PAYLOAD_PTR]] : i1, !llvm.ptr
  // CHECK: %[[TENSOR_VALUE:.*]] = llvm.load %[[SELECTED_TENSOR_PTR]] : !llvm.ptr -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  // CHECK-NOT: tvm_ffi.to
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[TENSOR_VALUE]] : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  // NO-CAST: return
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> !dlpack.tensor
  return %0 : !dlpack.tensor
}

// CHECK-LABEL: func.func @lowering_from_object
// CHECK-SAME: (%[[FROM_OBJECT_ARG:.*]]: !llvm.ptr) -> !llvm.struct<(i32, i32, i64)>
func.func @lowering_from_object(%h: !tvm_ffi.object_handle) -> !tvm_ffi.any {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[FROM_OBJECT_TYPE:.*]] = llvm.mlir.constant(70 : i32) : i32
  // CHECK: %[[FROM_OBJECT_BITS:.*]] = llvm.ptrtoint %[[FROM_OBJECT_ARG]] : !llvm.ptr to i64
  // CHECK: %[[FROM_OBJECT_ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[FROM_OBJECT_INIT:.*]] = llvm.mlir.poison : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[FROM_OBJECT_WITH_TYPE:.*]] = llvm.insertvalue %[[FROM_OBJECT_TYPE]], %[[FROM_OBJECT_INIT]][0]
  // CHECK: %[[FROM_OBJECT_WITH_AUX:.*]] = llvm.insertvalue %[[FROM_OBJECT_ZERO]], %[[FROM_OBJECT_WITH_TYPE]][1]
  // CHECK: %[[FROM_OBJECT_VALUE:.*]] = llvm.insertvalue %[[FROM_OBJECT_BITS]], %[[FROM_OBJECT_WITH_AUX]][2]
  // CHECK: return %[[FROM_OBJECT_VALUE]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.to %h : !tvm_ffi.object_handle -> !tvm_ffi.any
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
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> !tvm_ffi.object_handle
  return %0 : !tvm_ffi.object_handle
}

// CHECK-LABEL: func.func @lowering_tensor_from_llvm
// CHECK-SAME: (%[[TENSOR_FROM_LLVM_ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
func.func @lowering_tensor_from_llvm(%x: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>) -> !dlpack.tensor {
  // CHECK-NOT: dlpack.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[TENSOR_FROM_LLVM_ARG]] : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  %0 = dlpack.tensor_from_llvm %x : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)> -> !dlpack.tensor
  return %0 : !dlpack.tensor
}

// CHECK-LABEL: func.func @lowering_tensor_to_llvm
// CHECK-SAME: (%[[TENSOR_TO_LLVM_ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
func.func @lowering_tensor_to_llvm(%x: !dlpack.tensor) -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)> {
  // CHECK-NOT: dlpack.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[TENSOR_TO_LLVM_ARG]] : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  %0 = dlpack.tensor_to_llvm %x : !dlpack.tensor -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  return %0 : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
}

// CHECK-LABEL: func.func @lowering_env_tensor_alloc
// CHECK-SAME: () -> !llvm.ptr
func.func @lowering_env_tensor_alloc() -> !tvm_ffi.object_handle {
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[SHAPE_SIZE:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[SHAPE_SLOT:.*]] = llvm.alloca %[[SHAPE_SIZE]] x i64 : (i64) -> !llvm.ptr
  // CHECK: %[[SHAPE0_PTR:.*]] = llvm.getelementptr %[[SHAPE_SLOT]][0] : (!llvm.ptr) -> !llvm.ptr, i64
  // CHECK: %[[SHAPE0:.*]] = llvm.mlir.constant(16 : i64)
  // CHECK: llvm.store %[[SHAPE0]], %[[SHAPE0_PTR]] : i64, !llvm.ptr
  // CHECK: %[[SHAPE1_PTR:.*]] = llvm.getelementptr %[[SHAPE_SLOT]][1] : (!llvm.ptr) -> !llvm.ptr, i64
  // CHECK: %[[SHAPE1:.*]] = llvm.mlir.constant(32 : i64)
  // CHECK: llvm.store %[[SHAPE1]], %[[SHAPE1_PTR]] : i64, !llvm.ptr
  // CHECK: %[[DTYPE_CODE:.*]] = llvm.mlir.constant(2 : i8)
  // CHECK: %[[DTYPE_BITS:.*]] = llvm.mlir.constant(32 : i8)
  // CHECK: %[[DTYPE_LANES:.*]] = llvm.mlir.constant(1 : i16)
  // CHECK: %[[DTYPE_INIT:.*]] = llvm.mlir.poison : !llvm.struct<packed (i8, i8, i16)>
  // CHECK: %[[DTYPE_WITH_CODE:.*]] = llvm.insertvalue %[[DTYPE_CODE]], %[[DTYPE_INIT]][0] : !llvm.struct<packed (i8, i8, i16)>
  // CHECK: %[[DTYPE_WITH_BITS:.*]] = llvm.insertvalue %[[DTYPE_BITS]], %[[DTYPE_WITH_CODE]][1] : !llvm.struct<packed (i8, i8, i16)>
  // CHECK: %[[DTYPE:.*]] = llvm.insertvalue %[[DTYPE_LANES]], %[[DTYPE_WITH_BITS]][2] : !llvm.struct<packed (i8, i8, i16)>
  // CHECK: %[[NDIM:.*]] = llvm.mlir.constant(2 : i32)
  // CHECK: %[[HANDLE:.*]] = llvm.call @__libtriton_tvmffi_env_tensor_alloc(%[[DTYPE]], %[[NDIM]], %[[SHAPE_SLOT]]) : (!llvm.struct<packed (i8, i8, i16)>, i32, !llvm.ptr) -> !llvm.ptr
  // CHECK-NOT: tvm_ffi.env_tensor_alloc
  // CHECK: return %[[HANDLE]] : !llvm.ptr
  %h = tvm_ffi.env_tensor_alloc dtype = f32, shape = [16, 32] : !tvm_ffi.object_handle
  return %h : !tvm_ffi.object_handle
}

