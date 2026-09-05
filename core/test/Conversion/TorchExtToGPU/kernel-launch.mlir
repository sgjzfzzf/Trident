//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torchext-to-gpu --convert-torch-to-tvm-ffi --convert-tvm-ffi-to-func --convert-tvm-ffi-to-llvm --convert-dlpack-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK:      gpu.binary @kernel
// CHECK:      func.func @test_kernel_launch
// CHECK-SAME: %[[TENS_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SCALAR_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK-DAG:  %[[C32:[a-zA-Z0-9_]+]] = llvm.mlir.constant(32 : i64) : i64
// CHECK-DAG:  %[[C16:[a-zA-Z0-9_]+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG:  %[[C128:[a-zA-Z0-9_]+]] = llvm.mlir.constant(128 : i64) : i64
// CHECK-DAG:  %[[C1:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:  %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:  %[[SHMEM:[a-zA-Z0-9_]+]] = llvm.mlir.constant(16384 : i32) : i32
// CHECK-DAG:  %[[CUDA_DEVICE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:      %[[TENSOR_OBJECT:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENS_ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[TENSOR_OBJECT_PTR:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[TENSOR_OBJECT]] : i64 to !llvm.ptr
// CHECK:      %[[DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[TENSOR_OBJECT_PTR]][24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:      %[[DLTENSOR:[a-zA-Z0-9_]+]] = llvm.load %[[DLTENSOR_PTR]] : !llvm.ptr -> !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK:      %[[DATA_PTR:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[DLTENSOR]][0] : !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK:      %[[SCALAR_I64:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[SCALAR_ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[DEVICE_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[C1]] x i32 : (i64) -> !llvm.ptr
// CHECK:      llvm.call @aoti_torch_get_current_device_index(%[[DEVICE_SLOT]]) : (!llvm.ptr) -> i32
// CHECK:      %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = llvm.load %[[DEVICE_SLOT]] : !llvm.ptr -> i32
// CHECK:      %[[STREAM:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIEnvGetStream(%[[CUDA_DEVICE]], %[[DEVICE_INDEX]]) : (i32, i32) -> !llvm.ptr
// CHECK:      gpu.launch_func <%[[STREAM]] : !llvm.ptr> @kernel::@entry
// CHECK:      blocks in (%[[C32]], %[[C16]], %[[C1]])
// CHECK:      threads in (%[[C128]], %[[C1]], %[[C1]])
// CHECK:      dynamic_shared_memory_size %[[SHMEM]]
// CHECK:      args(%[[DATA_PTR]] : !llvm.ptr, %[[SCALAR_I64]] : i64, %[[ZERO]] : i64, %[[ZERO]] : i64)

// CHECK-NOT: torchext.trident_kernel_launch
module attributes { gpu.container_module } {
  gpu.binary @kernel [#gpu.object<#nvvm.target, "">]

  func.func @test_kernel_launch(
    %tensor: !torch.vtensor<[4],f32>,
    %scalar: !torch.int) {
    %c32 = llvm.mlir.constant(32 : i64) : i64
    %c16 = llvm.mlir.constant(16 : i64) : i64
    %c128 = llvm.mlir.constant(128 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %shmem = llvm.mlir.constant(16384 : i32) : i32

    torchext.trident_kernel_launch @kernel::@entry
      blocks in (%c32, %c16, %c1) : i64 threads in (%c128, %c1, %c1)
      dynamic_shared_memory_size %shmem
      args (%tensor : !torch.vtensor<[4],f32> {triton.specialization = #torchext.specialization<kind = !llvm.ptr, divisibility = 16>}, %scalar : !torch.int {triton.specialization = #torchext.specialization<kind = i64, divisibility = 16>})
    func.return
  }
}
