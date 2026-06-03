// RUN: libtriton-core-opt %s -convert-to-llvm | FileCheck %s
// RUN: libtriton-core-opt %s -convert-to-llvm | mlir-opt -convert-func-to-llvm
// -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir -o /dev/null

// CHECK-DAG: llvm.func @TVMFFIErrorSetRaisedFromCStr
// CHECK-DAG: llvm.func @TVMFFIObjectDecRef
// CHECK-DAG: llvm.func @TVMFFIObjectIncRef
// CHECK-DAG: llvm.func @TVMFFIEnvTensorAlloc
// CHECK-DAG: llvm.func @__libtriton_get_current_device

// CHECK-LABEL: func.func @lower_any_from_i64
// CHECK-SAME: (%[[FROM_INT_ARG:.*]]: i64) -> !llvm.struct<(i32, i32, i64)>
func.func @lower_any_from_i64(%i: i64) -> !tvm_ffi.any {
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

// CHECK-LABEL: func.func @lower_i64_from_any
// CHECK-SAME: (%[[TO_INT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> i64
func.func @lower_i64_from_any(%a: !tvm_ffi.any) -> i64 {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_INT_VALUE:.*]] = llvm.extractvalue %[[TO_INT_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: return %[[TO_INT_VALUE]] : i64
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> i64
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_any_from_i32
// CHECK-SAME: (%[[FROM_I32_ARG:.*]]: i32) -> !llvm.struct<(i32, i32, i64)>
func.func @lower_any_from_i32(%i: i32) -> !tvm_ffi.any {
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

// CHECK-LABEL: func.func @lower_any_from_llvm_struct
// CHECK-SAME: (%[[AS_FROM_LLVM_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
func.func @lower_any_from_llvm_struct(%a: !llvm.struct<(i32, i32, i64)>) -> !tvm_ffi.any {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[AS_FROM_LLVM_ARG]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.as %a : !llvm.struct<(i32, i32, i64)> -> !tvm_ffi.any
  return %0 : !tvm_ffi.any
}

// CHECK-LABEL: func.func @lower_llvm_struct_from_any
// CHECK-SAME: (%[[AS_TO_LLVM_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)>
func.func @lower_llvm_struct_from_any(%a: !tvm_ffi.any) -> !llvm.struct<(i32, i32, i64)> {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[AS_TO_LLVM_ARG]] : !llvm.struct<(i32, i32, i64)>
  %0 = tvm_ffi.as %a : !tvm_ffi.any -> !llvm.struct<(i32, i32, i64)>
  return %0 : !llvm.struct<(i32, i32, i64)>
}

// CHECK-LABEL: func.func @lower_any_from_f64
// CHECK-SAME: (%[[FROM_FLOAT_ARG:.*]]: f64) -> !llvm.struct<(i32, i32, i64)>
func.func @lower_any_from_f64(%f: f64) -> !tvm_ffi.any {
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

// CHECK-LABEL: func.func @lower_type_index_from_any
// CHECK-SAME: (%[[GET_TYPE_INDEX_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> i32
func.func @lower_type_index_from_any(%a: !tvm_ffi.any) -> i32 {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[GET_TYPE_INDEX_VALUE:.*]] = llvm.extractvalue %[[GET_TYPE_INDEX_ARG]][0] : !llvm.struct<(i32, i32, i64)>
  // CHECK: return %[[GET_TYPE_INDEX_VALUE]] : i32
  %0 = tvm_ffi.get_type_index %a : !tvm_ffi.any -> i32
  return %0 : i32
}

// CHECK-LABEL: func.func @lower_f64_from_any
// CHECK-SAME: (%[[TO_FLOAT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> f64
func.func @lower_f64_from_any(%a: !tvm_ffi.any) -> f64 {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_FLOAT_BITS:.*]] = llvm.extractvalue %[[TO_FLOAT_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[TO_FLOAT_VALUE:.*]] = llvm.bitcast %[[TO_FLOAT_BITS]] : i64 to f64
  // CHECK: return %[[TO_FLOAT_VALUE]] : f64
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> f64
  return %0 : f64
}

// CHECK-LABEL: func.func @lower_any_from_ptr
// CHECK-SAME: (%[[FROM_STR_ARG:.*]]: !llvm.ptr) -> !llvm.struct<(i32, i32, i64)>
func.func @lower_any_from_ptr(%p: !llvm.ptr) -> !tvm_ffi.any {
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

// CHECK-LABEL: func.func @lower_ptr_from_any
// CHECK-SAME: (%[[TO_STR_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr
func.func @lower_ptr_from_any(%a: !tvm_ffi.any) -> !llvm.ptr {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_STR_BITS:.*]] = llvm.extractvalue %[[TO_STR_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[TO_STR_VALUE:.*]] = llvm.inttoptr %[[TO_STR_BITS]] : i64 to !llvm.ptr
  // CHECK: return %[[TO_STR_VALUE]] : !llvm.ptr
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> !llvm.ptr
  return %0 : !llvm.ptr
}

// CHECK-LABEL: func.func @lower_error_set_from_c_str
// CHECK-SAME: (%[[ERROR_KIND:.*]]: !llvm.ptr, %[[ERROR_MESSAGE:.*]]: !llvm.ptr)
func.func @lower_error_set_from_c_str(%kind: !llvm.ptr, %message: !llvm.ptr) {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: llvm.call @TVMFFIErrorSetRaisedFromCStr(%[[ERROR_KIND]], %[[ERROR_MESSAGE]]) : (!llvm.ptr, !llvm.ptr) -> ()
  // CHECK: return
  tvm_ffi.error_set_raised_from_c_str %kind, %message : !llvm.ptr, !llvm.ptr
  return
}

// CHECK-LABEL: func.func @lower_object_handle_inc_ref
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @lower_object_handle_inc_ref(%obj: !tvm_ffi.object_handle) {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: llvm.call @TVMFFIObjectIncRef(%[[OBJ]]) : (!llvm.ptr) -> i32
  // CHECK: return
  tvm_ffi.object_inc_ref %obj : !tvm_ffi.object_handle
  return
}

// CHECK-LABEL: func.func @lower_object_handle_dec_ref
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @lower_object_handle_dec_ref(%obj: !tvm_ffi.object_handle) {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: llvm.call @TVMFFIObjectDecRef(%[[OBJ]]) : (!llvm.ptr) -> i32
  // CHECK: return
  tvm_ffi.object_dec_ref %obj : !tvm_ffi.object_handle
  return
}

// CHECK-LABEL: func.func @lower_tensor_from_any
// CHECK-SAME: (%[[TO_TENSOR_ARG:.*]]: !llvm.struct<(i32, i32, i64)>)
// CHECK-SAME: -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
// NO-CAST-LABEL: func.func @lower_tensor_from_any
func.func @lower_tensor_from_any(%a: !tvm_ffi.any) -> !dlpack.tensor {
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

// CHECK-LABEL: func.func @lower_any_from_object_handle
// CHECK-SAME: (%[[FROM_OBJECT_ARG:.*]]: !llvm.ptr) -> !llvm.struct<(i32, i32, i64)>
func.func @lower_any_from_object_handle(%h: !tvm_ffi.object_handle) -> !tvm_ffi.any {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[FROM_OBJECT_TENSOR_TYPE:.*]] = llvm.mlir.constant(70 : i32) : i32
  // CHECK: %[[FROM_OBJECT_NONE_TYPE:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[FROM_OBJECT_NULL:.*]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[FROM_OBJECT_IS_NULL:.*]] = llvm.icmp "eq" %[[FROM_OBJECT_ARG]], %[[FROM_OBJECT_NULL]] : !llvm.ptr
  // CHECK: %[[FROM_OBJECT_TYPE:.*]] = llvm.select %[[FROM_OBJECT_IS_NULL]], %[[FROM_OBJECT_NONE_TYPE]], %[[FROM_OBJECT_TENSOR_TYPE]] : i1, i32
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

// CHECK-LABEL: func.func @lower_object_handle_from_any
// CHECK-SAME: (%[[TO_OBJECT_ARG:.*]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.ptr
func.func @lower_object_handle_from_any(%a: !tvm_ffi.any) -> !tvm_ffi.object_handle {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_OBJECT_BITS:.*]] = llvm.extractvalue %[[TO_OBJECT_ARG]][2] : !llvm.struct<(i32, i32, i64)>
  // CHECK: %[[TO_OBJECT_VALUE:.*]] = llvm.inttoptr %[[TO_OBJECT_BITS]] : i64 to !llvm.ptr
  // CHECK: return %[[TO_OBJECT_VALUE]] : !llvm.ptr
  %0 = tvm_ffi.to %a : !tvm_ffi.any -> !tvm_ffi.object_handle
  return %0 : !tvm_ffi.object_handle
}

// CHECK-LABEL: func.func @lower_tensor_from_object_handle
// CHECK-SAME: (%[[TO_TENSOR_FROM_HANDLE_ARG:.*]]: !llvm.ptr)
// CHECK-SAME: -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
func.func @lower_tensor_from_object_handle(%h: !tvm_ffi.object_handle) -> !dlpack.tensor {
  // CHECK-NOT: tvm_ffi.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: %[[TO_TENSOR_FROM_HANDLE_PTR:.*]] = llvm.getelementptr %[[TO_TENSOR_FROM_HANDLE_ARG]][24] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK: %[[TO_TENSOR_FROM_HANDLE_VALUE:.*]] = llvm.load %[[TO_TENSOR_FROM_HANDLE_PTR]] : !llvm.ptr -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  // CHECK: return %[[TO_TENSOR_FROM_HANDLE_VALUE]] : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  %0 = tvm_ffi.to %h : !tvm_ffi.object_handle -> !dlpack.tensor
  return %0 : !dlpack.tensor
}

// CHECK-LABEL: func.func @lower_tensor_from_llvm_struct
// CHECK-SAME: (%[[TENSOR_FROM_LLVM_ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
func.func @lower_tensor_from_llvm_struct(%x: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>) -> !dlpack.tensor {
  // CHECK-NOT: dlpack.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[TENSOR_FROM_LLVM_ARG]] : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  %0 = dlpack.tensor_from_llvm %x : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)> -> !dlpack.tensor
  return %0 : !dlpack.tensor
}

// CHECK-LABEL: func.func @lower_llvm_struct_from_tensor
// CHECK-SAME: (%[[TENSOR_TO_LLVM_ARG:.*]]: !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>)
// CHECK-SAME: -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
func.func @lower_llvm_struct_from_tensor(%x: !dlpack.tensor) -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)> {
  // CHECK-NOT: dlpack.
  // CHECK-NOT: builtin.unrealized_conversion_cast
  // CHECK: return %[[TENSOR_TO_LLVM_ARG]] : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  %0 = dlpack.tensor_to_llvm %x : !dlpack.tensor -> !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  return %0 : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
}

// CHECK-LABEL: func.func @lower_object_handle_env_tensor_alloc
// CHECK-SAME: () -> !llvm.ptr
func.func @lower_object_handle_env_tensor_alloc() -> !tvm_ffi.object_handle {
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
  // CHECK: %[[DEVICE:.*]] = llvm.call @__libtriton_get_current_device() : () -> !llvm.struct<packed (i32, i32)>
  // CHECK: %[[TENSOR_INIT:.*]] = llvm.mlir.poison : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  // CHECK: %[[TENSOR_WITH_DEVICE:.*]] = llvm.insertvalue %[[DEVICE]], %{{.*}}[1] : !llvm.struct<packed (ptr, struct<packed (i32, i32)>, i32, struct<packed (i8, i8, i16)>, ptr, ptr, i64)>
  // CHECK: %[[OUT_SLOT:.*]] = llvm.alloca %{{.*}} x !llvm.ptr : (i64) -> !llvm.ptr
  // CHECK: llvm.call @TVMFFIEnvTensorAlloc(%{{.*}}, %[[OUT_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: %[[HANDLE:.*]] = llvm.load %[[OUT_SLOT]] : !llvm.ptr -> !llvm.ptr
  // CHECK-NOT: tvm_ffi.env_tensor_alloc
  // CHECK: return %[[HANDLE]] : !llvm.ptr
  %h = tvm_ffi.env_tensor_alloc dtype = f32, shape = [16, 32] : !tvm_ffi.object_handle
  return %h : !tvm_ffi.object_handle
}
