// RUN: libtriton-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @__tvm_ffi_void_func(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @void_func() {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[C42:.*]] = torch.constant.int 42
// CHECK-NEXT:    %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[C42]] : !torch.int to i64
// CHECK-NEXT:    %[[PAYLOAD:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    llvm.store %[[CAST]], %[[PAYLOAD]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @make_int() -> !torch.int {
  %0 = torch.constant.int 42
  tvm_ffi.return %0 : !torch.int
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_print_int(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : i64 to !torch.int
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @print_int(%arg0: !torch.int) {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_identity(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TO_TORCH:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : i64 to !torch.int
// CHECK-NEXT:    %[[TO_LLVM:.*]] = builtin.unrealized_conversion_cast %[[TO_TORCH]] : !torch.int to i64
// CHECK-NEXT:    %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    llvm.store %[[TO_LLVM]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @identity(%arg0: !torch.int) -> !torch.int {
  tvm_ffi.return %arg0 : !torch.int
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_identity_bool(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TRUNC:.*]] = llvm.trunc %[[RAW]] : i64 to i1
// CHECK-NEXT:    %[[TO_TORCH:.*]] = builtin.unrealized_conversion_cast %[[TRUNC]] : i1 to !torch.bool
// CHECK-NEXT:    %[[TO_I1:.*]] = builtin.unrealized_conversion_cast %[[TO_TORCH]] : !torch.bool to i1
// CHECK:         %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    %[[ZEXT:.*]] = llvm.zext %[[TO_I1]] : i1 to i64
// CHECK-NEXT:    llvm.store %[[ZEXT]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @identity_bool(%arg0: !torch.bool) -> !torch.bool {
  tvm_ffi.return %arg0 : !torch.bool
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_identity_float(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TO_F64:.*]] = llvm.bitcast %[[RAW]] : i64 to f64
// CHECK-NEXT:    %[[TO_TORCH:.*]] = builtin.unrealized_conversion_cast %[[TO_F64]] : f64 to !torch.float
// CHECK-NEXT:    %[[TO_F64_2:.*]] = builtin.unrealized_conversion_cast %[[TO_TORCH]] : !torch.float to f64
// CHECK:         %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    %[[TO_I64:.*]] = llvm.bitcast %[[TO_F64_2]] : f64 to i64
// CHECK-NEXT:    llvm.store %[[TO_I64]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @identity_float(%arg0: !torch.float) -> !torch.float {
  tvm_ffi.return %arg0 : !torch.float
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_add(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT0:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY0:.*]] = llvm.getelementptr %[[SLOT0]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW0:.*]] = llvm.load %[[PAY0]] : !llvm.ptr -> i64
// CHECK:         %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[RAW0]] : i64 to !torch.int
// CHECK:         %[[SLOT1:.*]] = llvm.getelementptr %[[ARGS_PTR]][1]
// CHECK:         %[[PAY1:.*]] = llvm.getelementptr %[[SLOT1]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW1:.*]] = llvm.load %[[PAY1]] : !llvm.ptr -> i64
// CHECK:         %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[RAW1]] : i64 to !torch.int
// CHECK:         %[[ADD:.*]] = torch.aten.add %[[ARG0]], %[[ARG1]] : !torch.int, !torch.int -> !torch.int
// CHECK:         %[[TO_I64:.*]] = builtin.unrealized_conversion_cast %[[ADD]] : !torch.int to i64
// CHECK-NEXT:    %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    llvm.store %[[TO_I64]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @add(%arg0: !torch.int, %arg1: !torch.int) -> !torch.int {
  %0 = torch.aten.add %arg0, %arg1 : !torch.int, !torch.int -> !torch.int
  tvm_ffi.return %0 : !torch.int
}

// -----

// Multi-block: a conditional branch using torch.prim.If.
// CHECK-LABEL: llvm.func @__tvm_ffi_cond_identity(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK:         %[[ARG:.*]] = builtin.unrealized_conversion_cast %[[RAW]] : i64 to !torch.int
// CHECK:         %[[GT:.*]] = torch.aten.gt.int %[[ARG]], %[[ARG]] : !torch.int, !torch.int -> !torch.bool
// CHECK:         %[[RESULT:.*]] = torch.prim.If %[[GT]] -> (!torch.int) {
// CHECK:         }
// CHECK:         %[[TO_I64:.*]] = builtin.unrealized_conversion_cast %[[RESULT]] : !torch.int to i64
// CHECK:         %[[RET_PAY:.*]] = llvm.getelementptr %[[RET_PTR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         llvm.store %[[TO_I64]], %[[RET_PAY]] : i64, !llvm.ptr
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.return %[[ZERO]] : i32
tvm_ffi.func @cond_identity(%arg0: !torch.int) -> !torch.int {
  %cond = torch.aten.gt.int %arg0, %arg0 : !torch.int, !torch.int -> !torch.bool
  %result = torch.prim.If %cond -> (!torch.int) {
    torch.prim.If.yield %arg0 : !torch.int
  } else {
    %zero = torch.constant.int 0
    torch.prim.If.yield %zero : !torch.int
  }
  tvm_ffi.return %result : !torch.int
}

// -----

// Tensor input: load DLTensor from TVMFFIAny and convert to AtenTensorHandle.
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_func(
// CHECK-SAME:    %{{.*}}: !llvm.ptr, %[[ARGS_PTR:.*]]: !llvm.ptr,
// CHECK-SAME:    %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ARGS_PTR]][0]
// CHECK-NEXT:    %[[PAY:.*]] = llvm.getelementptr %[[SLOT]]{{\[}}0, 2]
// CHECK-NEXT:    %[[DLTENSOR_I64:.*]] = llvm.load %[[PAY]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[DLTENSOR:.*]] = llvm.inttoptr %[[DLTENSOR_I64]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[DATA_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 0]
// CHECK-NEXT:    %[[DATA:.*]] = llvm.load %[[DATA_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[DEVTYPE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 1, 0]
// CHECK-NEXT:    %[[DL_DEV_TYPE:.*]] = llvm.load %[[DEVTYPE_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[DEVID_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 1, 1]
// CHECK-NEXT:    %[[DEVICE_ID:.*]] = llvm.load %[[DEVID_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[NDIM_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 2]
// CHECK-NEXT:    %[[NDIM:.*]] = llvm.load %[[NDIM_PTR]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[DTCODE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 3, 0]
// CHECK-NEXT:    %[[DTYPE_CODE:.*]] = llvm.load %[[DTCODE_PTR]] : !llvm.ptr -> i8
// CHECK-NEXT:    %[[DTBITS_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 3, 1]
// CHECK-NEXT:    %[[DTYPE_BITS:.*]] = llvm.load %[[DTBITS_PTR]] : !llvm.ptr -> i8
// CHECK-NEXT:    %[[SHAPE_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 4]
// CHECK-NEXT:    %[[SHAPE:.*]] = llvm.load %[[SHAPE_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[STRIDES_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 5]
// CHECK-NEXT:    %[[STRIDES:.*]] = llvm.load %[[STRIDES_PTR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[BOFF_PTR:.*]] = llvm.getelementptr %[[DLTENSOR]][0, 6]
// CHECK-NEXT:    %[[BYTE_OFF:.*]] = llvm.load %[[BOFF_PTR]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[TORCH_DTYPE:.*]] = llvm.call @mLibTritonTVMFFIToTorchType(%[[DTYPE_CODE]], %[[DTYPE_BITS]]) : (i8, i8) -> i32
// CHECK-NEXT:    %[[DEVICE_TYPE:.*]] = llvm.call @mLibTritonTVMFFIDeviceToTorchDeviceType(%[[DL_DEV_TYPE]]) : (i32) -> i32
// CHECK-NEXT:    %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[HANDLE_SLOT:.*]] = llvm.alloca %[[ONE]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[NDIM_I64:.*]] = llvm.sext %[[NDIM]] : i32 to i64
// CHECK-NEXT:    %[[RET:.*]] = llvm.call @aoti_torch_create_tensor_from_blob(%[[DATA]], %[[NDIM_I64]], %[[SHAPE]], %[[STRIDES]], %[[BYTE_OFF]], %[[TORCH_DTYPE]], %[[DEVICE_TYPE]], %[[DEVICE_ID]], %[[HANDLE_SLOT]]) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[HANDLE:.*]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[HANDLE]] : !llvm.ptr to !torch.tensor
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @tensor_func(%arg0: !torch.tensor) {
  tvm_ffi.return
}
