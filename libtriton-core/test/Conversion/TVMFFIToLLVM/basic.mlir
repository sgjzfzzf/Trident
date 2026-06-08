// RUN: libtriton-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s


// void_func:
// CHECK-LABEL: llvm.func @__tvm_ffi_void_func(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @void_func() {
  tvm_ffi.return
}

// -----

// make_int:
// CHECK-LABEL: llvm.func @__tvm_ffi_make_int(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    %int42 = torch.constant.int 42
// CHECK-NEXT:    [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast %int42 : !torch.int to i64
// CHECK-NEXT:    [[GEP:%[a-z0-9]+]] = llvm.getelementptr %arg3[0, 2]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    llvm.store [[CAST]], [[GEP]] : i64, !llvm.ptr
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @make_int() -> !torch.int {
  %0 = torch.constant.int 42
  tvm_ffi.return %0 : !torch.int
}

// -----

// print_int:
// CHECK-LABEL: llvm.func @__tvm_ffi_print_int(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[IDX_GEP:%[a-z0-9]+]] = llvm.getelementptr [[ARG_GEP]][0, 0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[TYPE_IDX:%[a-z0-9]+]] = llvm.load [[IDX_GEP]] : !llvm.ptr -> i32
// CHECK-NEXT:    [[EXPECTED:%[a-z0-9]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    [[MISMATCH:%[a-z0-9]+]] = llvm.icmp "ne" [[TYPE_IDX]], [[EXPECTED]] : i32
// CHECK-NEXT:    llvm.cond_br [[MISMATCH]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__libtriton_constant_kind_TypeError : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__libtriton_constant_msg_tvm_ffi: argument type mismatch" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[MINUS1:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[MINUS1]] : i32
// CHECK:       ^bb2:
// CHECK-NEXT:    [[PAYLOAD_GEP:%[a-z0-9]+]] = llvm.getelementptr [[ARG_GEP]][0, 2]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[PAYLOAD:%[a-z0-9]+]] = llvm.load [[PAYLOAD_GEP]] : !llvm.ptr -> i64
// CHECK-NEXT:    [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[PAYLOAD]] : i64 to !torch.int
// CHECK-NEXT:    llvm.br ^bb3
// CHECK:       ^bb3:
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @print_int(%arg0: !torch.int) {
  tvm_ffi.return
}

// -----

// identity_bool:
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_bool(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[IDX_GEP:%[a-z0-9]+]] = llvm.getelementptr [[ARG_GEP]][0, 0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[TYPE_IDX:%[a-z0-9]+]] = llvm.load [[IDX_GEP]] : !llvm.ptr -> i32
// CHECK-NEXT:    [[EXPECTED:%[a-z0-9]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    [[MISMATCH:%[a-z0-9]+]] = llvm.icmp "ne" [[TYPE_IDX]], [[EXPECTED]] : i32
// CHECK-NEXT:    llvm.cond_br [[MISMATCH]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__libtriton_constant_kind_TypeError : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__libtriton_constant_msg_tvm_ffi: argument type mismatch" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[MINUS1:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[MINUS1]] : i32
// CHECK:       ^bb2:
// CHECK-NEXT:    [[PAYLOAD_GEP:%[a-z0-9]+]] = llvm.getelementptr [[ARG_GEP]][0, 2]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[PAYLOAD:%[a-z0-9]+]] = llvm.load [[PAYLOAD_GEP]] : !llvm.ptr -> i64
// CHECK-NEXT:    [[TRUNC:%[a-z0-9]+]] = llvm.trunc [[PAYLOAD]] : i64 to i1
// CHECK-NEXT:    [[ARG_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[TRUNC]] : i1 to !torch.bool
// CHECK-NEXT:    llvm.br ^bb3
// CHECK:       ^bb3:
// CHECK-NEXT:    [[RET_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[ARG_CAST]] : !torch.bool to i1
// CHECK-NEXT:    [[RET_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg3[0, 2]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[EXT:%[a-z0-9]+]] = llvm.zext [[RET_CAST]] : i1 to i64
// CHECK-NEXT:    llvm.store [[EXT]], [[RET_GEP]] : i64, !llvm.ptr
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @identity_bool(%arg0: !torch.bool) -> !torch.bool {
  tvm_ffi.return %arg0 : !torch.bool
}

// -----

// identity_float:
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_float(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[IDX_GEP:%[a-z0-9]+]] = llvm.getelementptr [[ARG_GEP]][0, 0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[TYPE_IDX:%[a-z0-9]+]] = llvm.load [[IDX_GEP]] : !llvm.ptr -> i32
// CHECK-NEXT:    [[EXPECTED:%[a-z0-9]+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT:    [[MISMATCH:%[a-z0-9]+]] = llvm.icmp "ne" [[TYPE_IDX]], [[EXPECTED]] : i32
// CHECK-NEXT:    llvm.cond_br [[MISMATCH]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__libtriton_constant_kind_TypeError : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__libtriton_constant_msg_tvm_ffi: argument type mismatch" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    [[MINUS1:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    llvm.return [[MINUS1]] : i32
// CHECK:       ^bb2:
// CHECK-NEXT:    [[PAYLOAD_GEP:%[a-z0-9]+]] = llvm.getelementptr [[ARG_GEP]][0, 2]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[PAYLOAD:%[a-z0-9]+]] = llvm.load [[PAYLOAD_GEP]] : !llvm.ptr -> i64
// CHECK-NEXT:    [[BC:%[a-z0-9]+]] = llvm.bitcast [[PAYLOAD]] : i64 to f64
// CHECK-NEXT:    [[ARG_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[BC]] : f64 to !torch.float
// CHECK-NEXT:    llvm.br ^bb3
// CHECK:       ^bb3:
// CHECK-NEXT:    [[RET_CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[ARG_CAST]] : !torch.float to f64
// CHECK-NEXT:    [[RET_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg3[0, 2]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[RET_BC:%[a-z0-9]+]] = llvm.bitcast [[RET_CAST]] : f64 to i64
// CHECK-NEXT:    llvm.store [[RET_BC]], [[RET_GEP]] : i64, !llvm.ptr
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @identity_float(%arg0: !torch.float) -> !torch.float {
  tvm_ffi.return %arg0 : !torch.float
}

// -----

// tensor_func:
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_func(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i32, i32, i64)>
// CHECK-NEXT:    [[C1:%[a-z0-9]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    [[SLOT:%[a-z0-9]+]] = llvm.alloca [[C1]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    [[RET:%[a-z0-9]+]] = llvm.call @mLibTritonUnpackTVMFFIAnyToTensor([[ARG_GEP]], [[SLOT]])
// CHECK-SAME:      : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    [[MINUS1:%[a-z0-9]+]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT:    [[MISMATCH:%[a-z0-9]+]] = llvm.icmp "eq" [[RET]], [[MINUS1]] : i32
// CHECK-NEXT:    llvm.cond_br [[MISMATCH]], ^bb1, ^bb2
// CHECK:       ^bb1:
// CHECK-NEXT:    [[KIND:%[a-z0-9]+]] = llvm.mlir.addressof @__libtriton_constant_kind_TypeError : !llvm.ptr
// CHECK-NEXT:    [[MSG:%[a-z0-9]+]] = llvm.mlir.addressof @"__libtriton_constant_msg_tvm_ffi: argument type mismatch" : !llvm.ptr
// CHECK-NEXT:    llvm.call @TVMFFIErrorSetRaisedFromCStr([[KIND]], [[MSG]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return [[MINUS1]] : i32
// CHECK:       ^bb2:
// CHECK-NEXT:    [[HANDLE:%[a-z0-9]+]] = llvm.load [[SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[HANDLE]] : !llvm.ptr to !torch.tensor
// CHECK-NEXT:    llvm.br ^bb3
// CHECK:       ^bb3:
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @tensor_func(%arg0: !torch.tensor) {
  tvm_ffi.return
}

// -----

// make_tensor:
// CHECK-LABEL: llvm.func @__tvm_ffi_make_tensor(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK-NEXT:    %int3 = torch.constant.int 3
// CHECK-NEXT:    %int4 = torch.constant.int 4
// CHECK-NEXT:    [[LIST:%[a-z0-9]+]] = torch.prim.ListConstruct %int3, %int4
// CHECK-SAME:      : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-NEXT:    %none = torch.constant.none
// CHECK-NEXT:    [[TENSOR:%[a-z0-9]+]] = torch.aten.empty.memory_format [[LIST]], %none, %none, %none, %none, %none
// CHECK-SAME:      : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor
// CHECK-NEXT:    [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[TENSOR]] : !torch.tensor to !llvm.ptr
// CHECK-NEXT:    [[RET:%[a-z0-9]+]] = llvm.call @mLibTritonPackTensorToTVMFFIAny([[CAST]], %arg3)
// CHECK-SAME:      : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @make_tensor() -> !torch.tensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %shape = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %tensor = torch.aten.empty.memory_format %shape, %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor
  tvm_ffi.return %tensor : !torch.tensor
}
