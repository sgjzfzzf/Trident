//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file | FileCheck %s --check-prefix=DIALECT
// RUN: sed -n '/^\/\/ BEGIN-GUARD-LOWERING/,$p' %s | trident-core-opt -split-input-file -convert-tvm-ffi-to-func -convert-scf-to-cf -convert-tvm-ffi-to-llvm -convert-arith-to-llvm -convert-cf-to-llvm -convert-func-to-llvm | FileCheck %s --check-prefix=LOWER

// DIALECT-LABEL: tvm_ffi.func @test() {
// DIALECT-NEXT:    tvm_ffi.return
// DIALECT-NEXT:  }
tvm_ffi.func @test() {
  tvm_ffi.return
}

// -----

// DIALECT-LABEL: tvm_ffi.func @with_torch_int(
// DIALECT-SAME:    [[INT_ARG:%[a-zA-Z0-9_]+]]: !torch.int) -> !torch.int {
// DIALECT-NEXT:    tvm_ffi.return [[INT_ARG]] : !torch.int
// DIALECT-NEXT:  }
tvm_ffi.func @with_torch_int(%arg0: !torch.int) -> !torch.int {
  tvm_ffi.return %arg0 : !torch.int
}

// -----

// DIALECT-LABEL: tvm_ffi.func @lifetime_types(
// DIALECT-SAME: [[TENSOR_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.tensor,
// DIALECT-SAME: [[ARRAY_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.array,
// DIALECT-SAME: [[ANY_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.any,
// DIALECT-SAME: [[INDEX_ARG:%[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.tensor {
// DIALECT:      [[ITEM:%[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item [[ARRAY_ARG]][[[INDEX_ARG]]] as !tvm_ffi.tensor : !tvm_ffi.array, !tvm_ffi.int -> !tvm_ffi.tensor
// DIALECT-NEXT: tvm_ffi.ObjectIncRef [[ITEM]] : !tvm_ffi.tensor
// DIALECT-NEXT: tvm_ffi.ObjectDecRef [[ARRAY_ARG]] : !tvm_ffi.array
// DIALECT-NEXT: tvm_ffi.ObjectIncRef [[ANY_ARG]] : !tvm_ffi.any
// DIALECT-NEXT: tvm_ffi.ObjectDecRef [[ANY_ARG]] : !tvm_ffi.any
// DIALECT-NEXT: tvm_ffi.return [[ITEM]] : !tvm_ffi.tensor
// DIALECT-NEXT: }
tvm_ffi.func @lifetime_types(
    %tensor: !tvm_ffi.tensor,
    %array: !tvm_ffi.array,
    %any: !tvm_ffi.any,
    %index: !tvm_ffi.int) -> !tvm_ffi.tensor {
  %item = tvm_ffi.array.get_item %array[%index]
      as !tvm_ffi.tensor
      : !tvm_ffi.array, !tvm_ffi.int
      -> !tvm_ffi.tensor
  tvm_ffi.ObjectIncRef %item : !tvm_ffi.tensor
  tvm_ffi.ObjectDecRef %array : !tvm_ffi.array
  tvm_ffi.ObjectIncRef %any : !tvm_ffi.any
  tvm_ffi.ObjectDecRef %any : !tvm_ffi.any
  tvm_ffi.return %item : !tvm_ffi.tensor
}

// DIALECT-LABEL: tvm_ffi.func @array_unifies_list_and_tuple(
// DIALECT-SAME: [[ARRAY_VALUE:%[a-zA-Z0-9_]+]]: !tvm_ffi.array,
// DIALECT-SAME: [[ARRAY_INDEX:%[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.int {
// DIALECT-NEXT: [[ARRAY_ITEM:%[a-zA-Z0-9_]+]] = tvm_ffi.array.get_item [[ARRAY_VALUE]][[[ARRAY_INDEX]]] as !tvm_ffi.int : !tvm_ffi.array, !tvm_ffi.int -> !tvm_ffi.int
// DIALECT-NEXT: tvm_ffi.return [[ARRAY_ITEM]] : !tvm_ffi.int
// DIALECT-NEXT: }
tvm_ffi.func @array_unifies_list_and_tuple(
    %list_or_tuple: !tvm_ffi.array,
    %index: !tvm_ffi.int) -> !tvm_ffi.int {
  %item = tvm_ffi.array.get_item %list_or_tuple[%index]
      as !tvm_ffi.int
      : !tvm_ffi.array, !tvm_ffi.int
      -> !tvm_ffi.int
  tvm_ffi.return %item : !tvm_ffi.int
}

// -----

// DIALECT-LABEL: tvm_ffi.func @function_call() -> !tvm_ffi.array {
// DIALECT-NEXT:    [[ARRAY_FUNC:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
// DIALECT-NEXT:    [[ARRAY_RESULT:%[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall [[ARRAY_FUNC]]() : () -> !tvm_ffi.array
// DIALECT-NEXT:    tvm_ffi.return [[ARRAY_RESULT]] : !tvm_ffi.array
// DIALECT-NEXT:  }
tvm_ffi.func @function_call() -> !tvm_ffi.array {
  %func = tvm_ffi.FunctionGetGlobal "ffi.Array" : !tvm_ffi.function
  %result = tvm_ffi.FunctionCall %func() : () -> !tvm_ffi.array
  tvm_ffi.return %result : !tvm_ffi.array
}

// -----

// A semantic TVM FFI value may be returned through the generic Any ABI slot.
// DIALECT-LABEL: tvm_ffi.func @any_return() -> !tvm_ffi.any {
// DIALECT-NEXT:    [[ANY_VALUE:%[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 1 : i64}> : () -> !tvm_ffi.int
// DIALECT-NEXT:    tvm_ffi.return [[ANY_VALUE]] : !tvm_ffi.int
// DIALECT-NEXT:  }
tvm_ffi.func @any_return() -> !tvm_ffi.any {
  %value = "tvm_ffi.constant"() <{value = 1 : i64}>
      : () -> !tvm_ffi.int
  tvm_ffi.return %value : !tvm_ffi.int
}

// -----

// BEGIN-GUARD-LOWERING

// LOWER-LABEL: llvm.func @tensor_guard(
// LOWER-SAME: [[TENSOR_ANY:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// LOWER: [[DIM_HANDLE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TENSOR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[DIM_OBJECT:%[a-zA-Z0-9_]+]] = llvm.inttoptr [[DIM_HANDLE]] : i64 to !llvm.ptr
// LOWER: [[DIM_TENSOR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DIM_OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// LOWER: [[DIM_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DIM_TENSOR]][0, 2]
// LOWER: [[DIM32:%[a-zA-Z0-9_]+]] = llvm.load [[DIM_PTR]] : !llvm.ptr -> i32
// LOWER: [[DIM:%[a-zA-Z0-9_]+]] = llvm.sext [[DIM32]] : i32 to i64
// LOWER: [[ZERO_INDEX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i64) : i64
// LOWER: [[SIZE_HANDLE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TENSOR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[SIZE_OBJECT:%[a-zA-Z0-9_]+]] = llvm.inttoptr [[SIZE_HANDLE]] : i64 to !llvm.ptr
// LOWER: [[SIZE_TENSOR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[SIZE_OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// LOWER: [[SHAPE_PTR_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[SIZE_TENSOR]][0, 4]
// LOWER: [[SHAPE_PTR:%[a-zA-Z0-9_]+]] = llvm.load [[SHAPE_PTR_PTR]] : !llvm.ptr -> !llvm.ptr
// LOWER: [[SIZE_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[SHAPE_PTR]][[[ZERO_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// LOWER: [[SIZE:%[a-zA-Z0-9_]+]] = llvm.load [[SIZE_PTR]] : !llvm.ptr -> i64
// LOWER: [[STRIDE_HANDLE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TENSOR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[STRIDE_OBJECT:%[a-zA-Z0-9_]+]] = llvm.inttoptr [[STRIDE_HANDLE]] : i64 to !llvm.ptr
// LOWER: [[STRIDE_TENSOR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[STRIDE_OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// LOWER: [[STRIDE_PTR_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[STRIDE_TENSOR]][0, 5]
// LOWER: [[STRIDE_PTR:%[a-zA-Z0-9_]+]] = llvm.load [[STRIDE_PTR_PTR]] : !llvm.ptr -> !llvm.ptr
// LOWER: [[STRIDE_VALUE_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[STRIDE_PTR]][[[ZERO_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// LOWER: [[STRIDE:%[a-zA-Z0-9_]+]] = llvm.load [[STRIDE_VALUE_PTR]] : !llvm.ptr -> i64
// LOWER: [[OFFSET_HANDLE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TENSOR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[OFFSET_OBJECT:%[a-zA-Z0-9_]+]] = llvm.inttoptr [[OFFSET_HANDLE]] : i64 to !llvm.ptr
// LOWER: [[OFFSET_TENSOR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[OFFSET_OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// LOWER: [[OFFSET_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[OFFSET_TENSOR]][0, 6]
// LOWER: [[OFFSET:%[a-zA-Z0-9_]+]] = llvm.load [[OFFSET_PTR]] : !llvm.ptr -> i64
// LOWER: [[DTYPE_HANDLE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TENSOR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[DTYPE_OBJECT:%[a-zA-Z0-9_]+]] = llvm.inttoptr [[DTYPE_HANDLE]] : i64 to !llvm.ptr
// LOWER: [[DTYPE_TENSOR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DTYPE_OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// LOWER: [[DTYPE_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DTYPE_TENSOR]][0, 3]
// LOWER: [[CODE_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DTYPE_PTR]][0, 0]
// LOWER: [[CODE:%[a-zA-Z0-9_]+]] = llvm.load [[CODE_PTR]] : !llvm.ptr -> i8
// LOWER: [[BITS_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DTYPE_PTR]][0, 1]
// LOWER: [[BITS:%[a-zA-Z0-9_]+]] = llvm.load [[BITS_PTR]] : !llvm.ptr -> i8
// LOWER: [[LANES_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DTYPE_PTR]][0, 2]
// LOWER: [[LANES:%[a-zA-Z0-9_]+]] = llvm.load [[LANES_PTR]] : !llvm.ptr -> i16
// LOWER: [[DEVICE_HANDLE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TENSOR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[DEVICE_OBJECT:%[a-zA-Z0-9_]+]] = llvm.inttoptr [[DEVICE_HANDLE]] : i64 to !llvm.ptr
// LOWER: [[DEVICE_TENSOR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DEVICE_OBJECT]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// LOWER: [[DEVICE_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DEVICE_TENSOR]][0, 1]
// LOWER: [[DEVICE_TYPE_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DEVICE_PTR]][0, 0]
// LOWER: [[DEVICE_TYPE:%[a-zA-Z0-9_]+]] = llvm.load [[DEVICE_TYPE_PTR]] : !llvm.ptr -> i32
// LOWER: [[DEVICE_INDEX_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[DEVICE_PTR]][0, 1]
// LOWER: [[DEVICE_INDEX:%[a-zA-Z0-9_]+]] = llvm.load [[DEVICE_INDEX_PTR]] : !llvm.ptr -> i32
// LOWER-NEXT: llvm.return

// LOWER-LABEL: llvm.func @__tvm_ffi_tensor_guard(
// LOWER-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// LOWER: [[TENSOR_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// LOWER: [[TENSOR_VALUE:%[a-zA-Z0-9_]+]] = llvm.load [[TENSOR_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// LOWER: [[TENSOR_TYPE_INDEX:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[TENSOR_VALUE]][0] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[TENSOR_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(70 : i32) : i32
// LOWER: [[IS_TENSOR:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[TENSOR_TYPE_INDEX]], [[TENSOR_KIND]] : i32
// LOWER: llvm.cond_br [[IS_TENSOR]], [[TENSOR_VALID:\^bb[0-9]+]], [[TENSOR_INVALID:\^bb[0-9]+]]
// LOWER: [[TENSOR_VALID]]:
// LOWER: llvm.call @tensor_guard([[TENSOR_VALUE]]) : (!llvm.struct<(i32, i32, i64)>) -> ()
tvm_ffi.func @tensor_guard(%arg0: !torch.tensor) attributes {emit_tvm_ffi_abi} {
  %dim = tvm_ffi.tensor.dim %arg0 : !torch.tensor
  %index = arith.constant 0 : i64
  %size = tvm_ffi.tensor.size %arg0[%index] : !torch.tensor
  %stride = tvm_ffi.tensor.stride %arg0[%index] : !torch.tensor
  %offset = tvm_ffi.tensor.storage_offset %arg0 : !torch.tensor
  %code, %bits, %lanes = tvm_ffi.tensor.dtype %arg0 : !torch.tensor
  %device_type, %device_index = tvm_ffi.tensor.device %arg0 : !torch.tensor
  tvm_ffi.return
}

// -----

// LOWER-LABEL: llvm.func @scalar_guard(
// LOWER-SAME: [[SCALAR_ANY:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// LOWER: [[EXPECTED_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[EXPECTED_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// LOWER: [[EXPECTED_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// LOWER: [[EXPECTED_WITH_KIND:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[EXPECTED_KIND]], [[EXPECTED_UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[ZERO_AUX:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// LOWER: [[EXPECTED_WITH_AUX:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[ZERO_AUX]], [[EXPECTED_WITH_KIND]][1] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[EXPECTED_ANY:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[EXPECTED_PAYLOAD]], [[EXPECTED_WITH_AUX]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[LHS_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[SCALAR_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[RHS_TYPE:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[EXPECTED_ANY]][0] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[SAME_TYPE:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[LHS_TYPE]], [[RHS_TYPE]] : i32
// LOWER: [[LHS_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[SCALAR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[RHS_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[EXPECTED_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[SAME_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[LHS_PAYLOAD]], [[RHS_PAYLOAD]] : i64
// LOWER: [[SAME_SCALAR:%[a-zA-Z0-9_]+]] = llvm.and [[SAME_TYPE]], [[SAME_PAYLOAD]] : i1
// LOWER-NEXT: llvm.return

// LOWER-LABEL: llvm.func @__tvm_ffi_scalar_guard(
// LOWER-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// LOWER: [[SCALAR_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// LOWER: [[SCALAR_VALUE:%[a-zA-Z0-9_]+]] = llvm.load [[SCALAR_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// LOWER: [[SCALAR_TYPE_INDEX:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[SCALAR_VALUE]][0] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[INT_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// LOWER: [[IS_INT:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[SCALAR_TYPE_INDEX]], [[INT_KIND]] : i32
// LOWER: llvm.cond_br [[IS_INT]], [[SCALAR_VALID:\^bb[0-9]+]], [[SCALAR_INVALID:\^bb[0-9]+]]
// LOWER: [[SCALAR_VALID]]:
// LOWER: llvm.call @scalar_guard([[SCALAR_VALUE]]) : (!llvm.struct<(i32, i32, i64)>) -> ()
tvm_ffi.func @scalar_guard(%arg0: !torch.int) attributes {emit_tvm_ffi_abi} {
  %expected = "tvm_ffi.constant"() <{value = 1 : i64}> : () -> !tvm_ffi.int
  %same = tvm_ffi.eq %arg0, %expected : !torch.int, !tvm_ffi.int
  tvm_ffi.return
}

// -----

// LOWER-LABEL: llvm.func @array_length(
// LOWER-SAME: [[ARRAY_ANY:%[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) {
// LOWER: [[ONE_ELEMENT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[SOURCE_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[ONE_ELEMENT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// LOWER: llvm.store [[ARRAY_ANY]], [[SOURCE_SLOT]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// LOWER: [[ONE_ARGUMENT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[CALL_ARGS:%[a-zA-Z0-9_]+]] = llvm.alloca [[ONE_ARGUMENT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// LOWER: [[CALL_ARG0:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[CALL_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// LOWER: [[ARRAY_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[SOURCE_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// LOWER: llvm.store [[ARRAY_COPY]], [[CALL_ARG0]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// LOWER: [[ARG_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// LOWER: [[ARRAY_SIZE_NAME:%[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_constant_ffi.ArraySize_ffi.ArraySize : !llvm.ptr
// LOWER: [[NAME_RECORD_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[FUNCTION_NAME:%[a-zA-Z0-9_]+]] = llvm.alloca [[NAME_RECORD_COUNT]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// LOWER: [[FUNCTION_NAME_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[FUNCTION_NAME]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// LOWER: llvm.store [[ARRAY_SIZE_NAME]], [[FUNCTION_NAME_PTR]] : !llvm.ptr, !llvm.ptr
// LOWER: [[NAME_LENGTH:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(13 : i64) : i64
// LOWER: [[NAME_LENGTH_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[FUNCTION_NAME]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// LOWER: llvm.store [[NAME_LENGTH]], [[NAME_LENGTH_PTR]] : i64, !llvm.ptr
// LOWER: [[HANDLE_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[FUNCTION_HANDLE_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[HANDLE_COUNT]] x !llvm.ptr : (i64) -> !llvm.ptr
// LOWER: [[GET_GLOBAL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal([[FUNCTION_NAME]], [[FUNCTION_HANDLE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// LOWER: [[FUNCTION_HANDLE:%[a-zA-Z0-9_]+]] = llvm.load [[FUNCTION_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// LOWER: [[RETURN_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[RETURN_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[RETURN_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// LOWER: [[CALL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall([[FUNCTION_HANDLE]], [[CALL_ARGS]], [[ARG_COUNT]], [[RETURN_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// LOWER: [[DEC_REF_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIObjectDecRef([[FUNCTION_HANDLE]]) : (!llvm.ptr) -> i32
// LOWER: [[LENGTH_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[RETURN_SLOT]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// LOWER: [[ARRAY_LENGTH:%[a-zA-Z0-9_]+]] = llvm.load [[LENGTH_PTR]] : !llvm.ptr -> i64
// LOWER-NEXT: llvm.return

// LOWER-LABEL: llvm.func @__tvm_ffi_array_length(
// LOWER-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// LOWER: [[ARRAY_SLOT:%[a-zA-Z0-9_]+]] = llvm.getelementptr %arg1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// LOWER: [[ARRAY_VALUE:%[a-zA-Z0-9_]+]] = llvm.load [[ARRAY_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// LOWER: [[ARRAY_TYPE_INDEX:%[a-zA-Z0-9_]+]] = llvm.extractvalue [[ARRAY_VALUE]][0] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[ARRAY_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(71 : i32) : i32
// LOWER: [[IS_ARRAY:%[a-zA-Z0-9_]+]] = llvm.icmp "eq" [[ARRAY_TYPE_INDEX]], [[ARRAY_KIND]] : i32
// LOWER: llvm.cond_br [[IS_ARRAY]], [[ARRAY_VALID:\^bb[0-9]+]], [[ARRAY_INVALID:\^bb[0-9]+]]
// LOWER: [[ARRAY_VALID]]:
// LOWER: llvm.call @array_length([[ARRAY_VALUE]]) : (!llvm.struct<(i32, i32, i64)>) -> ()
tvm_ffi.func @array_length(%arg0: !tvm_ffi.array) attributes {emit_tvm_ffi_abi} {
  %length = tvm_ffi.array.length %arg0 : !tvm_ffi.array
  tvm_ffi.return
}

// -----

// LOWER-LABEL: llvm.func @exception_value() -> !llvm.struct<(i32, i32, i64)> {
// LOWER: [[KIND_NAME:%[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_constant_ExceptionKind_GuardMatch : !llvm.ptr
// LOWER: [[EXCEPTION_UNDEF:%[a-zA-Z0-9_]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// LOWER: [[STRING_KIND:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(8 : i32) : i32
// LOWER: [[EXCEPTION_WITH_KIND:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[STRING_KIND]], [[EXCEPTION_UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[ZERO_AUXILIARY:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i32) : i32
// LOWER: [[EXCEPTION_WITH_AUX:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[ZERO_AUXILIARY]], [[EXCEPTION_WITH_KIND]][1] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[KIND_PAYLOAD:%[a-zA-Z0-9_]+]] = llvm.ptrtoint [[KIND_NAME]] : !llvm.ptr to i64
// LOWER: [[EXCEPTION_ARGUMENT:%[a-zA-Z0-9_]+]] = llvm.insertvalue [[KIND_PAYLOAD]], [[EXCEPTION_WITH_AUX]][2] : !llvm.struct<(i32, i32, i64)>
// LOWER: [[EXCEPTION_ARG_COUNT64:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[EXCEPTION_ARG_SOURCE:%[a-zA-Z0-9_]+]] = llvm.alloca [[EXCEPTION_ARG_COUNT64]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// LOWER: llvm.store [[EXCEPTION_ARGUMENT]], [[EXCEPTION_ARG_SOURCE]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// LOWER: [[EXCEPTION_CALL_COUNT64:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[EXCEPTION_CALL_ARGS:%[a-zA-Z0-9_]+]] = llvm.alloca [[EXCEPTION_CALL_COUNT64]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// LOWER: [[EXCEPTION_CALL_ARG0:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[EXCEPTION_CALL_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// LOWER: [[EXCEPTION_ARG_COPY:%[a-zA-Z0-9_]+]] = llvm.load [[EXCEPTION_ARG_SOURCE]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// LOWER: llvm.store [[EXCEPTION_ARG_COPY]], [[EXCEPTION_CALL_ARG0]] : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// LOWER: [[EXCEPTION_ARG_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32) : i32
// LOWER: [[EXCEPTION_FUNC_NAME:%[a-zA-Z0-9_]+]] = llvm.mlir.addressof @__trident_constant_trident.ffi.Exception_trident.ffi.Exception : !llvm.ptr
// LOWER: [[EXCEPTION_NAME_RECORDS:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[EXCEPTION_NAME_BUFFER:%[a-zA-Z0-9_]+]] = llvm.alloca [[EXCEPTION_NAME_RECORDS]] x !llvm.struct<(ptr, i64)> : (i64) -> !llvm.ptr
// LOWER: [[EXCEPTION_NAME_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[EXCEPTION_NAME_BUFFER]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// LOWER: llvm.store [[EXCEPTION_FUNC_NAME]], [[EXCEPTION_NAME_PTR]] : !llvm.ptr, !llvm.ptr
// LOWER: [[EXCEPTION_NAME_LENGTH:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(21 : i64) : i64
// LOWER: [[EXCEPTION_NAME_LENGTH_PTR:%[a-zA-Z0-9_]+]] = llvm.getelementptr [[EXCEPTION_NAME_BUFFER]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64)>
// LOWER: llvm.store [[EXCEPTION_NAME_LENGTH]], [[EXCEPTION_NAME_LENGTH_PTR]] : i64, !llvm.ptr
// LOWER: [[EXCEPTION_HANDLE_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[EXCEPTION_HANDLE_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[EXCEPTION_HANDLE_COUNT]] x !llvm.ptr : (i64) -> !llvm.ptr
// LOWER: [[EXCEPTION_GET_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionGetGlobal([[EXCEPTION_NAME_BUFFER]], [[EXCEPTION_HANDLE_SLOT]]) : (!llvm.ptr, !llvm.ptr) -> i32
// LOWER: [[EXCEPTION_HANDLE:%[a-zA-Z0-9_]+]] = llvm.load [[EXCEPTION_HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// LOWER: [[EXCEPTION_RESULT_COUNT:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// LOWER: [[EXCEPTION_RESULT_SLOT:%[a-zA-Z0-9_]+]] = llvm.alloca [[EXCEPTION_RESULT_COUNT]] x !llvm.struct<(i32, i32, i64)> : (i64) -> !llvm.ptr
// LOWER: [[EXCEPTION_CALL_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIFunctionCall([[EXCEPTION_HANDLE]], [[EXCEPTION_CALL_ARGS]], [[EXCEPTION_ARG_COUNT]], [[EXCEPTION_RESULT_SLOT]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// LOWER: [[EXCEPTION_DEC_REF_STATUS:%[a-zA-Z0-9_]+]] = llvm.call @TVMFFIObjectDecRef([[EXCEPTION_HANDLE]]) : (!llvm.ptr) -> i32
// LOWER: [[EXCEPTION_RESULT:%[a-zA-Z0-9_]+]] = llvm.load [[EXCEPTION_RESULT_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// LOWER: llvm.return [[EXCEPTION_RESULT]] : !llvm.struct<(i32, i32, i64)>

// LOWER-LABEL: llvm.func @__tvm_ffi_exception_value(
// LOWER-SAME: %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// LOWER: [[ALWAYS_TRUE:%[a-zA-Z0-9_]+]] = llvm.mlir.constant(true) : i1
// LOWER: llvm.cond_br [[ALWAYS_TRUE]], [[EXCEPTION_VALID:\^bb[0-9]+]], [[EXCEPTION_INVALID:\^bb[0-9]+]]
// LOWER: [[EXCEPTION_VALID]]:
// LOWER: [[SEMANTIC_EXCEPTION:%[a-zA-Z0-9_]+]] = llvm.call @exception_value() : () -> !llvm.struct<(i32, i32, i64)>
// LOWER: [[TVM_EXCEPTION:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[SEMANTIC_EXCEPTION]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.exception
// LOWER: [[ABI_EXCEPTION:%[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast [[TVM_EXCEPTION]] : !tvm_ffi.exception to !llvm.struct<(i32, i32, i64)>
// LOWER: llvm.store [[ABI_EXCEPTION]], %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
tvm_ffi.func @exception_value() -> !tvm_ffi.exception attributes {emit_tvm_ffi_abi} {
  %exception = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  tvm_ffi.return %exception : !tvm_ffi.exception
}
