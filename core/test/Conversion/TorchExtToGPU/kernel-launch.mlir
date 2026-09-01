//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torch-to-tvm-ffi --convert-torchext-to-gpu --convert-tvm-ffi-to-func --convert-tvm-ffi-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK:      llvm.func @TVMFFIEnvGetStream(i32, i32) -> !llvm.ptr
// CHECK:      llvm.func @aoti_torch_get_current_device_index(!llvm.ptr) -> i32
// CHECK:      gpu.module @kernel {
// CHECK:        gpu.func @entry(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64, %arg3: i64)
// CHECK-SAME:   kernel attributes {gpu.binary = ""}
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:   }
// CHECK:      func.func @test_kernel_launch
// CHECK-SAME: %[[TENS_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>, %[[SCAL_ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[C32:[a-zA-Z0-9_]+]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:      %[[C16:[a-zA-Z0-9_]+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:      %[[C128:[a-zA-Z0-9_]+]] = llvm.mlir.constant(128 : i64) : i64
// CHECK:      %[[C1:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:      %[[SHMEM:[a-zA-Z0-9_]+]] = llvm.mlir.constant(16384 : i32) : i32
// CHECK:      %[[HANDLE_I64:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[TENS_ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[HANDLE:[a-zA-Z0-9_]+]] = llvm.inttoptr %[[HANDLE_I64]] : i64 to !llvm.ptr
// CHECK:      %[[DLTENSOR_PTR:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[HANDLE]]{{\[}}24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:      %[[DATA_GEP:[a-zA-Z0-9_]+]] = llvm.getelementptr %[[DLTENSOR_PTR]]{{\[}}0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK:      %[[DATA_PTR:[a-zA-Z0-9_]+]] = llvm.load %[[DATA_GEP]] : !llvm.ptr -> !llvm.ptr
// CHECK:      %[[SCALAR_PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[SCAL_ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[ONE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:      %[[DEV_IDX_SLOT:[a-zA-Z0-9_]+]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// CHECK:      llvm.call @aoti_torch_get_current_device_index(%[[DEV_IDX_SLOT]]) : (!llvm.ptr) -> i32
// CHECK:      %[[DEVICE_IDX:[a-zA-Z0-9_]+]] = llvm.load %[[DEV_IDX_SLOT]] : !llvm.ptr -> i32
// CHECK:      %[[CUDA_TYPE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:      %[[ASYNC_OBJ:[a-zA-Z0-9_]+]] = llvm.call @TVMFFIEnvGetStream(%[[CUDA_TYPE]], %[[DEVICE_IDX]]) : (i32, i32) -> !llvm.ptr
// CHECK:      %[[ZERO:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:      gpu.launch_func <%[[ASYNC_OBJ]] : !llvm.ptr> @kernel::@entry
// CHECK:      blocks in (%[[C32]], %[[C16]], %[[C1]])
// CHECK:      threads in (%[[C128]], %[[C1]], %[[C1]])
// CHECK:      dynamic_shared_memory_size %[[SHMEM]]
// CHECK:      args(%[[DATA_PTR]] : !llvm.ptr, %[[SCALAR_PLD]] : i64, %[[ZERO]] : i64, %[[ZERO]] : i64)

// CHECK-NOT: torchext.trident_kernel_launch
module attributes { gpu.container_module } {
  gpu.module @kernel {
    gpu.func @entry(%arg0: !llvm.ptr, %arg1: i64, %arg2: i64, %arg3: i64)
        attributes { gpu.binary = "", gpu.kernel } {
      gpu.return
    }
  }

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
      args (%tensor, %scalar : !torch.vtensor<[4],f32>, !torch.int)
    func.return
  }
}
