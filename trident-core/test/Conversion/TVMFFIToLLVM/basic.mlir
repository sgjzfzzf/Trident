//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -split-input-file -convert-tvm-ffi-to-llvm | FileCheck %s


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
// CHECK:         llvm.store {{%.*}}, %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
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
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[LOADED:%[a-z0-9]+]] = llvm.load [[ARG_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.int
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @print_int(%arg0: !torch.int) {
  tvm_ffi.return
}

// -----

// identity_bool:
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_bool(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[LOADED:%[a-z0-9]+]] = llvm.load [[ARG_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.bool
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK:         llvm.store {{%.*}}, %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @identity_bool(%arg0: !torch.bool) -> !torch.bool {
  tvm_ffi.return %arg0 : !torch.bool
}

// -----

// identity_float:
// CHECK-LABEL: llvm.func @__tvm_ffi_identity_float(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[LOADED:%[a-z0-9]+]] = llvm.load [[ARG_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.float
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK:         llvm.store {{%.*}}, %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @identity_float(%arg0: !torch.float) -> !torch.float {
  tvm_ffi.return %arg0 : !torch.float
}

// -----

// tensor_func:
// CHECK-LABEL: llvm.func @__tvm_ffi_tensor_func(
// CHECK-SAME:      %arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i32, %arg3: !llvm.ptr) -> i32 {
// CHECK:         [[ARG_GEP:%[a-z0-9]+]] = llvm.getelementptr %arg1[0]
// CHECK-SAME:      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[LOADED:%[a-z0-9]+]] = llvm.load [[ARG_GEP]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK-NEXT:    [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[LOADED]] : !llvm.struct<(i32, i32, i64)> to !torch.tensor
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:  ^bb1:
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
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
// CHECK-NEXT:    [[TENSOR:%[a-z0-9]+]] = torch.aten.empty.memory_format [[LIST]]
// CHECK:         llvm.store {{%.*}}, %arg3 : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:         [[ZERO:%[a-z0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return [[ZERO]] : i32
tvm_ffi.func @make_tensor() -> !torch.tensor {
  %int3 = torch.constant.int 3
  %int4 = torch.constant.int 4
  %shape = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %tensor = torch.aten.empty.memory_format %shape, %none, %none, %none, %none, %none : !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.tensor
  tvm_ffi.return %tensor : !torch.tensor
}
