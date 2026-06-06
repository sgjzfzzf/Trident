// RUN: libtriton-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @__tvm_ffi_void_func(
// CHECK-SAME:    %[[ABI_ARGS:.*]]: !llvm.ptr, %[[TYPE_CODES:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK-NEXT:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[ZERO]] : i32
tvm_ffi.func @void_func() {
  tvm_ffi.return
}

// -----

// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK-SAME:    %[[ABI_ARGS:.*]]: !llvm.ptr, %[[TYPE_CODES:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
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
// CHECK-SAME:    %[[ABI_ARGS:.*]]: !llvm.ptr, %[[TYPE_CODES:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ABI_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
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
// CHECK-SAME:    %[[ABI_ARGS:.*]]: !llvm.ptr, %[[TYPE_CODES:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ABI_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
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
// CHECK-SAME:    %[[ABI_ARGS:.*]]: !llvm.ptr, %[[TYPE_CODES:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ABI_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
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
// CHECK-SAME:    %[[ABI_ARGS:.*]]: !llvm.ptr, %[[TYPE_CODES:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ABI_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
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
// CHECK-SAME:    %[[ABI_ARGS:.*]]: !llvm.ptr, %[[TYPE_CODES:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT0:.*]] = llvm.getelementptr %[[ABI_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[PAY0:.*]] = llvm.getelementptr %[[SLOT0]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK:         %[[RAW0:.*]] = llvm.load %[[PAY0]] : !llvm.ptr -> i64
// CHECK:         %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[RAW0]] : i64 to !torch.int
// CHECK:         %[[SLOT1:.*]] = llvm.getelementptr %[[ABI_ARGS]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
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
// CHECK-SAME:    %[[ABI_ARGS:.*]]: !llvm.ptr, %[[TYPE_CODES:.*]]: !llvm.ptr, %[[NUM_ARGS:.*]]: i32, %[[RET_PTR:.*]]: !llvm.ptr
// CHECK:         %[[SLOT:.*]] = llvm.getelementptr %[[ABI_ARGS]][0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
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
