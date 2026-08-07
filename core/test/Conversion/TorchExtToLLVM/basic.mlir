//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --trident-lowering-pipeline -split-input-file | FileCheck %s
//
// Tests that the TorchExt dialect ops are lowered to LLVM when the
// --trident-lowering-pipeline runs.  TorchExt ops are handled by the
// TorchExtToGPU pass (torchext.cast scalar extraction + torchext.trident_kernel_launch
// -> gpu.launch_func); this test exercises them end-to-end through the full
// pipeline.

// CHECK-LABEL: llvm.func @cast_float_to_f32
// CHECK-SAME:  %[[ARG:.*]]: !llvm.struct<(i32, i32, i64)>
// The scalar payload is extracted from the TVMFFIAny struct, bitcast to f64
// and truncated to f32.
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[F64:.*]] = llvm.bitcast %[[PLD]] : i64 to f64
// CHECK:       %[[F32:.*]] = llvm.fptrunc %[[F64]] : f64 to f32
// CHECK:       llvm.return %[[F32]] : f32
func.func @cast_float_to_f32(%arg0: !torch.float) -> f32 {
  %0 = torchext.cast %arg0 : !torch.float -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: llvm.func @cast_int_to_i64
// CHECK-SAME:  %[[ARG:.*]]: !llvm.struct<(i32, i32, i64)>
// An !torch.int -> i64 cast is a plain payload extraction (no truncation).
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NOT:   llvm.trunc
// CHECK:       llvm.return %[[PLD]] : i64
func.func @cast_int_to_i64(%arg0: !torch.int) -> i64 {
  %0 = torchext.cast %arg0 : !torch.int -> i64
  return %0 : i64
}

// -----

// CHECK-LABEL: llvm.func @test_kernel_launch
// CHECK-SAME:  (%[[TENS:.*]]: !llvm.struct<(i32, i32, i64)>, %[[SCAL:.*]]: !llvm.struct<(i32, i32, i64)>) {
// The tensor data pointer is obtained by walking the DLTensor header...
// CHECK:       %[[HDL:.*]] = llvm.extractvalue %[[TENS]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PTR:.*]] = llvm.inttoptr %[[HDL]] : i64 to !llvm.ptr
// CHECK:       llvm.getelementptr %[[PTR]]{{.*}}[24]
// ...and the scalar payload is passed as the second kernel argument.
// CHECK:       %[[SPLD:.*]] = llvm.extractvalue %[[SCAL]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.call @TVMFFIEnvGetStream
// CHECK:       gpu.launch_func <%{{.*}} : !llvm.ptr> @kernel::@entry
// CHECK-NOT:   torchext.
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
      blocks in (%c32, %c16, %c1) threads in (%c128, %c1, %c1)
      dynamic_shared_memory_size %shmem
      args (%tensor, %scalar : !torch.vtensor<[4],f32>, !torch.int)
    func.return
  }
}

