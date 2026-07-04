//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @__tvm_ffi_constant_guard_bool_false(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.bool
// CHECK:         [[TYPE_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 0]
// CHECK:         [[TYPE_VAL:%[a-z0-9]+]] = llvm.load [[TYPE_GEP]]
// CHECK:         [[PAYLOAD_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[PAYLOAD_VAL:%[a-z0-9]+]] = llvm.load [[PAYLOAD_GEP]]
// CHECK:         [[EXPECTED_TYPE:%[a-z0-9]+]] = llvm.mlir.constant(2 : i32)
// CHECK:         [[EXPECTED_PAYLOAD:%[a-z0-9]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         [[TYPE_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[TYPE_VAL]], [[EXPECTED_TYPE]]
// CHECK:         [[PAYLOAD_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[PAYLOAD_VAL]], [[EXPECTED_PAYLOAD]]
// CHECK:         [[ALL:%[a-z0-9]+]] = llvm.and [[TYPE_CMP]], [[PAYLOAD_CMP]]
// CHECK-NEXT:    llvm.cond_br [[ALL]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
tvm_ffi.func @constant_guard_bool_false(%arg0: !torch.bool {tvm_ffi.guard = [#tvm_ffi.ConstantGuard<type_index = 2, payload = 0>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_constant_guard_float_half(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.float
// CHECK:         [[TYPE_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 0]
// CHECK:         [[TYPE_VAL:%[a-z0-9]+]] = llvm.load [[TYPE_GEP]]
// CHECK:         [[PAYLOAD_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[PAYLOAD_VAL:%[a-z0-9]+]] = llvm.load [[PAYLOAD_GEP]]
// CHECK:         [[EXPECTED_TYPE:%[a-z0-9]+]] = llvm.mlir.constant(3 : i32)
// CHECK:         [[EXPECTED_PAYLOAD:%[a-z0-9]+]] = llvm.mlir.constant(4602678819172646912 : i64) : i64
// CHECK:         [[TYPE_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[TYPE_VAL]], [[EXPECTED_TYPE]]
// CHECK:         [[PAYLOAD_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[PAYLOAD_VAL]], [[EXPECTED_PAYLOAD]]
// CHECK:         [[ALL:%[a-z0-9]+]] = llvm.and [[TYPE_CMP]], [[PAYLOAD_CMP]]
// CHECK-NEXT:    llvm.cond_br [[ALL]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
tvm_ffi.func @constant_guard_float_half(%arg0: !torch.float {tvm_ffi.guard = [#tvm_ffi.ConstantGuard<type_index = 3, payload = 4602678819172646912>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_constant_guard_int_42(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.int
// CHECK:         [[TYPE_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 0]
// CHECK:         [[TYPE_VAL:%[a-z0-9]+]] = llvm.load [[TYPE_GEP]]
// CHECK:         [[PAYLOAD_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[PAYLOAD_VAL:%[a-z0-9]+]] = llvm.load [[PAYLOAD_GEP]]
// CHECK:         [[EXPECTED_TYPE:%[a-z0-9]+]] = llvm.mlir.constant(1 : i32)
// CHECK:         [[EXPECTED_PAYLOAD:%[a-z0-9]+]] = llvm.mlir.constant(42 : i64) : i64
// CHECK:         [[TYPE_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[TYPE_VAL]], [[EXPECTED_TYPE]]
// CHECK:         [[PAYLOAD_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[PAYLOAD_VAL]], [[EXPECTED_PAYLOAD]]
// CHECK:         [[ALL:%[a-z0-9]+]] = llvm.and [[TYPE_CMP]], [[PAYLOAD_CMP]]
// CHECK-NEXT:    llvm.cond_br [[ALL]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
tvm_ffi.func @constant_guard_int_42(%arg0: !torch.int {tvm_ffi.guard = [#tvm_ffi.ConstantGuard<type_index = 1, payload = 42>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_cuda_device_guard(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK:         [[RAW_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[RAW_I64:%[a-z0-9]+]] = llvm.load [[RAW_GEP]]
// CHECK:         [[OBJ:%[a-z0-9]+]] = llvm.inttoptr [[RAW_I64]]
// CHECK:         [[TENSOR:%[a-z0-9]+]] = llvm.getelementptr [[OBJ]][24]
// CHECK:         [[DEV_TYPE_GEP:%[a-z0-9]+]] = llvm.getelementptr [[TENSOR]][0, 1, 0]
// CHECK:         [[DEV_TYPE:%[a-z0-9]+]] = llvm.load [[DEV_TYPE_GEP]]
// CHECK:         [[DEV_ID_GEP:%[a-z0-9]+]] = llvm.getelementptr [[TENSOR]][0, 1, 1]
// CHECK:         [[DEV_ID:%[a-z0-9]+]] = llvm.load [[DEV_ID_GEP]]
// CHECK:         [[TYPE_CVAL:%[a-z0-9]+]] = llvm.mlir.constant(2 : i32)
// CHECK:         [[TYPE_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[DEV_TYPE]], [[TYPE_CVAL]]
// CHECK:         [[ID_CVAL:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32)
// CHECK:         [[ID_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[DEV_ID]], [[ID_CVAL]]
// CHECK:         [[AND:%[a-z0-9]+]] = llvm.and [[TYPE_CMP]], [[ID_CMP]]
// CHECK-NEXT:    llvm.cond_br [[AND]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK-NEXT:  ^bb{{[0-9]+}}:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__trident_constant_GuardMatchExceptionKind_GuardMatchException : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__trident_constant_GuardMatchExceptionMsg_argument 0 fails guard check" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[ERR:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[ERR]] : i32
tvm_ffi.func @cuda_device_guard(%arg0: !torch.tensor {tvm_ffi.guard = [#tvm_ffi.CudaDeviceGuard<device_type = 2, device_index = 0>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_dimension_guard(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK:         [[RAW_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[RAW_I64:%[a-z0-9]+]] = llvm.load [[RAW_GEP]]
// CHECK:         [[OBJ:%[a-z0-9]+]] = llvm.inttoptr [[RAW_I64]]
// CHECK:         [[TENSOR:%[a-z0-9]+]] = llvm.getelementptr [[OBJ]][24]
// CHECK:         [[NDIM_GEP:%[a-z0-9]+]] = llvm.getelementptr [[TENSOR]][0, 2]
// CHECK:         [[NDIM:%[a-z0-9]+]] = llvm.load [[NDIM_GEP]]
// CHECK:         [[EXPECTED:%[a-z0-9]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         [[CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[NDIM]], [[EXPECTED]]
// CHECK-NEXT:    llvm.cond_br [[CMP]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK-NEXT:  ^bb{{[0-9]+}}:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__trident_constant_GuardMatchExceptionKind_GuardMatchException : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__trident_constant_GuardMatchExceptionMsg_argument 0 fails guard check" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[ERR:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[ERR]] : i32
tvm_ffi.func @dimension_guard(%arg0: !torch.tensor {tvm_ffi.guard = [#tvm_ffi.DimensionGuard<expected = 2>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_dtype_guard_f16(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[RAW_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[RAW_I64:%[a-z0-9]+]] = llvm.load [[RAW_GEP]]
// CHECK:         [[OBJ:%[a-z0-9]+]] = llvm.inttoptr [[RAW_I64]]
// CHECK:         [[TENSOR:%[a-z0-9]+]] = llvm.getelementptr [[OBJ]][24]
// CHECK:         [[DTYPE_GEP:%[a-z0-9]+]] = llvm.getelementptr [[TENSOR]][0, 3]
// CHECK:         [[BITS_GEP:%[a-z0-9]+]] = llvm.getelementptr [[DTYPE_GEP]][0, 1]
// CHECK:         [[BITS:%[a-z0-9]+]] = llvm.load [[BITS_GEP]]
// CHECK:         [[EXP_BITS:%[a-z0-9]+]] = llvm.mlir.constant(16 : i8)
// CHECK:         llvm.icmp "eq" [[BITS]], [[EXP_BITS]]
tvm_ffi.func @dtype_guard_f16(%arg0: !torch.tensor {tvm_ffi.guard = [#tvm_ffi.DtypeGuard<code = 2, bits = 16, lanes = 1>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_dtype_guard_f32(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK:         [[RAW_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[RAW_I64:%[a-z0-9]+]] = llvm.load [[RAW_GEP]]
// CHECK:         [[OBJ:%[a-z0-9]+]] = llvm.inttoptr [[RAW_I64]]
// CHECK:         [[TENSOR:%[a-z0-9]+]] = llvm.getelementptr [[OBJ]][24]
// CHECK:         [[DTYPE_GEP:%[a-z0-9]+]] = llvm.getelementptr [[TENSOR]][0, 3]
// CHECK:         [[CODE_GEP:%[a-z0-9]+]] = llvm.getelementptr [[DTYPE_GEP]][0, 0]
// CHECK:         [[BITS_GEP:%[a-z0-9]+]] = llvm.getelementptr [[DTYPE_GEP]][0, 1]
// CHECK:         [[LANES_GEP:%[a-z0-9]+]] = llvm.getelementptr [[DTYPE_GEP]][0, 2]
// CHECK:         [[CODE:%[a-z0-9]+]] = llvm.load [[CODE_GEP]]
// CHECK:         [[BITS:%[a-z0-9]+]] = llvm.load [[BITS_GEP]]
// CHECK:         [[LANES:%[a-z0-9]+]] = llvm.load [[LANES_GEP]]
// CHECK:         [[EXP_CODE:%[a-z0-9]+]] = llvm.mlir.constant(2 : i8)
// CHECK:         [[EXP_BITS:%[a-z0-9]+]] = llvm.mlir.constant(32 : i8)
// CHECK:         [[EXP_LANES:%[a-z0-9]+]] = llvm.mlir.constant(1 : i16)
// CHECK:         [[CODE_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[CODE]], [[EXP_CODE]]
// CHECK:         [[BITS_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[BITS]], [[EXP_BITS]]
// CHECK:         [[LANES_CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[LANES]], [[EXP_LANES]]
// CHECK:         [[CODE_BITS:%[a-z0-9]+]] = llvm.and [[CODE_CMP]], [[BITS_CMP]]
// CHECK:         [[ALL:%[a-z0-9]+]] = llvm.and [[CODE_BITS]], [[LANES_CMP]]
// CHECK-NEXT:    llvm.cond_br [[ALL]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK-NEXT:  ^bb{{[0-9]+}}:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__trident_constant_GuardMatchExceptionKind_GuardMatchException : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__trident_constant_GuardMatchExceptionMsg_argument 0 fails guard check" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[ERR:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[ERR]] : i32
tvm_ffi.func @dtype_guard_f32(%arg0: !torch.tensor {tvm_ffi.guard = [#tvm_ffi.DtypeGuard<code = 2, bits = 32, lanes = 1>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_size_guard(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK:         [[RAW_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[RAW_I64:%[a-z0-9]+]] = llvm.load [[RAW_GEP]]
// CHECK:         [[OBJ:%[a-z0-9]+]] = llvm.inttoptr [[RAW_I64]]
// CHECK:         [[TENSOR:%[a-z0-9]+]] = llvm.getelementptr [[OBJ]][24]
// CHECK:         [[SHAPE_PTR_GEP:%[a-z0-9]+]] = llvm.getelementptr [[TENSOR]][0, 4]
// CHECK:         [[SHAPE_PTR:%[a-z0-9]+]] = llvm.load [[SHAPE_PTR_GEP]]
// CHECK:         [[SHAPE_ELEM_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SHAPE_PTR]][0]
// CHECK:         [[SHAPE_ELEM:%[a-z0-9]+]] = llvm.load [[SHAPE_ELEM_GEP]]
// CHECK:         [[EXP:%[a-z0-9]+]] = llvm.mlir.constant(64 : i64) : i64
// CHECK:         [[CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[SHAPE_ELEM]], [[EXP]]
// CHECK-NEXT:    llvm.cond_br [[CMP]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK-NEXT:  ^bb{{[0-9]+}}:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__trident_constant_GuardMatchExceptionKind_GuardMatchException : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__trident_constant_GuardMatchExceptionMsg_argument 0 fails guard check" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[ERR:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[ERR]] : i32
tvm_ffi.func @size_guard(%arg0: !torch.tensor {tvm_ffi.guard = [#tvm_ffi.SizeGuard<index = 0, expected = 64>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_storage_offset_guard(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK:         [[RAW_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[RAW_I64:%[a-z0-9]+]] = llvm.load [[RAW_GEP]]
// CHECK:         [[OBJ:%[a-z0-9]+]] = llvm.inttoptr [[RAW_I64]]
// CHECK:         [[TENSOR:%[a-z0-9]+]] = llvm.getelementptr [[OBJ]][24]
// CHECK:         [[OFFSET_GEP:%[a-z0-9]+]] = llvm.getelementptr [[TENSOR]][0, 6]
// CHECK:         [[OFFSET:%[a-z0-9]+]] = llvm.load [[OFFSET_GEP]]
// CHECK:         [[EXP:%[a-z0-9]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         [[CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[OFFSET]], [[EXP]]
// CHECK-NEXT:    llvm.cond_br [[CMP]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK-NEXT:  ^bb{{[0-9]+}}:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__trident_constant_GuardMatchExceptionKind_GuardMatchException : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__trident_constant_GuardMatchExceptionMsg_argument 0 fails guard check" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[ERR:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[ERR]] : i32
tvm_ffi.func @storage_offset_guard(%arg0: !torch.tensor {tvm_ffi.guard = [#tvm_ffi.StorageOffsetGuard<expected = 0>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_stride_guard(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK:         [[RAW_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 2]
// CHECK:         [[RAW_I64:%[a-z0-9]+]] = llvm.load [[RAW_GEP]]
// CHECK:         [[OBJ:%[a-z0-9]+]] = llvm.inttoptr [[RAW_I64]]
// CHECK:         [[TENSOR:%[a-z0-9]+]] = llvm.getelementptr [[OBJ]][24]
// CHECK:         [[STRIDE_PTR_GEP:%[a-z0-9]+]] = llvm.getelementptr [[TENSOR]][0, 5]
// CHECK:         [[STRIDE_PTR:%[a-z0-9]+]] = llvm.load [[STRIDE_PTR_GEP]]
// CHECK:         [[STRIDE_ELEM_GEP:%[a-z0-9]+]] = llvm.getelementptr [[STRIDE_PTR]][1]
// CHECK:         [[STRIDE_ELEM:%[a-z0-9]+]] = llvm.load [[STRIDE_ELEM_GEP]]
// CHECK:         [[EXP:%[a-z0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:         [[CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[STRIDE_ELEM]], [[EXP]]
// CHECK-NEXT:    llvm.cond_br [[CMP]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK-NEXT:  ^bb{{[0-9]+}}:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__trident_constant_GuardMatchExceptionKind_GuardMatchException : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__trident_constant_GuardMatchExceptionMsg_argument 0 fails guard check" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[ERR:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[ERR]] : i32
tvm_ffi.func @stride_guard(%arg0: !torch.tensor {tvm_ffi.guard = [#tvm_ffi.StrideGuard<index = 1, expected = 1>]}) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_type_guard(
// CHECK:         [[SLOT:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK:         [[LOADED:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[SLOT_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK:         [[TYPE_GEP:%[a-z0-9]+]] = llvm.getelementptr [[SLOT]][0, 0]
// CHECK:         [[TYPE_CODE:%[a-z0-9]+]] = llvm.load [[TYPE_GEP]]
// CHECK:         [[EXPECTED:%[a-z0-9]+]] = llvm.mlir.constant(70 : i32) : i32
// CHECK:         [[CMP:%[a-z0-9]+]] = llvm.icmp "eq" [[TYPE_CODE]], [[EXPECTED]]
// CHECK-NEXT:    llvm.cond_br [[CMP]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK-NEXT:  ^bb{{[0-9]+}}:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__trident_constant_GuardMatchExceptionKind_GuardMatchException : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__trident_constant_GuardMatchExceptionMsg_argument 0 fails guard check" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[ERR:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[ERR]] : i32
tvm_ffi.func @tensor_type_guard(%arg0: !torch.tensor {tvm_ffi.guard = [#tvm_ffi.TensorTypeGuard<>]}) {
  tvm_ffi.return
}
