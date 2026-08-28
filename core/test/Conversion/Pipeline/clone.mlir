//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline | FileCheck %s

// A concrete torch.aten.clone is generalized before any folding pass and
// reaches the name-based TVM FFI dispatch path with its memory format operand.
// In particular, the contiguous-memory-format clone must not fold to its input:
// the Python regression test passes a transposed tensor here and checks that a
// distinct, contiguous allocation is returned.

// CHECK-DAG: llvm.func @TVMFFIFunctionGetGlobal(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @TVMFFIFunctionCall(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG: llvm.mlir.global internal constant @__trident_constant_trident.aten.clone_trident.aten.clone("trident.aten.clone\00")
// CHECK-LABEL: llvm.func @torch.aten.clone(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>) -> !llvm.struct<(i32, i32, i64)> {
// CHECK: %[[NARGS:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[TWO:[0-9]+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: %[[ZERO_I32:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[ZERO_I64:[0-9]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: %[[FORMAT_UNDEF:[0-9]+]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK: %[[INT_TYPE_CODE:[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[FORMAT_WITH_TYPE:[0-9]+]] = llvm.insertvalue %[[INT_TYPE_CODE]], %[[FORMAT_UNDEF]][0]
// CHECK: %[[FORMAT_WITH_DEVICE:[0-9]+]] = llvm.insertvalue %[[ZERO_I32]], %[[FORMAT_WITH_TYPE]][1]
// CHECK: %[[FORMAT:[0-9]+]] = llvm.insertvalue %[[ZERO_I64]], %[[FORMAT_WITH_DEVICE]][2]
// CHECK: %[[GETGLOBAL:[0-9]+]] = llvm.call @TVMFFIFunctionGetGlobal(%[[FUNCTION_NAME:[0-9]+]], %[[HANDLE_SLOT:[0-9]+]])
// CHECK: %[[HANDLE:[0-9]+]] = llvm.load %[[HANDLE_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK: %[[ARGS:[0-9]+]] = llvm.alloca %[[TWO]] x !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.store %[[ARG0]], %[[ARGS]]
// CHECK: %[[FORMAT_SLOT:[0-9]+]] = llvm.getelementptr %[[ARGS]][1]
// CHECK: llvm.store %[[FORMAT]], %[[FORMAT_SLOT]]
// CHECK: %[[RET_SLOT:[0-9]+]] = llvm.alloca
// CHECK: %[[CALL:[0-9]+]] = llvm.call @TVMFFIFunctionCall(%[[HANDLE]], %[[CALL_ARGS:[0-9]+]], %[[NARGS]], %[[RET_SLOT]])
// CHECK: llvm.call @TVMFFIObjectDecRef(%[[HANDLE]])
// CHECK: %[[RET:[0-9]+]] = llvm.load %[[RET_SLOT]] : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: llvm.return %[[RET]] : !llvm.struct<(i32, i32, i64)>
// CHECK-LABEL: llvm.func @__tvm_ffi_clone(
// CHECK: %[[WRAP_ARG:[0-9]+]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[WRAP_RET:[0-9]+]] = llvm.call @clone(%[[WRAP_ARG]])
// CHECK: llvm.store %[[WRAP_RET]], %arg3
// CHECK-LABEL: llvm.func @__tvm_ffi_clone_preserve(
// CHECK: %[[PRESERVE_ARG:[0-9]+]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[PRESERVE_RET:[0-9]+]] = llvm.call @clone_preserve(%[[PRESERVE_ARG]])
// CHECK: llvm.store %[[PRESERVE_RET]], %arg3

func.func @torch.aten.clone(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> {
  %memory_format = torch.constant.int 0
  %clone = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  return %clone : !torch.vtensor<[32,2],f32>
}

tvm_ffi.func @clone(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> attributes {emit_tvm_ffi_abi} {
  %memory_format = torch.constant.int 0
  %clone = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  tvm_ffi.return %clone : !torch.vtensor<[32,2],f32>
}

// Preserve-format clone is covered separately so the runtime test can verify
// that the memory-format operand is not merely present but also honored.
tvm_ffi.func @clone_preserve(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> attributes {emit_tvm_ffi_abi} {
  %memory_format = torch.constant.int 1
  %clone = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  tvm_ffi.return %clone : !torch.vtensor<[32,2],f32>
}
