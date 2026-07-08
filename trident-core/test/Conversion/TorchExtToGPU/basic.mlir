//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torchext-to-gpu -split-input-file | FileCheck %s

// CHECK-LABEL: module attributes {gpu.container_module}
// CHECK:      llvm.func @aoti_torch_get_current_stream(i32, !llvm.ptr) -> i32
// CHECK:      llvm.func @aoti_torch_get_current_device_index(!llvm.ptr) -> i32
// CHECK:      gpu.module @kernel {
// CHECK:        gpu.func @entry(%{{.*}}: !llvm.ptr, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64)
// CHECK-SAME:   kernel attributes {gpu.binary = ""}
// CHECK-NEXT:     gpu.return
// CHECK-NEXT:   }
// CHECK:      func.func @test_kernel_launch
// CHECK-SAME: %[[TENS_ARG:.*]]: !torch.vtensor<[4],f32>
// CHECK-SAME: %[[SCAL_ARG:.*]]: !torch.int
// CHECK:      %[[SCALAR_ANY:.*]] = builtin.unrealized_conversion_cast %[[SCAL_ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[TENSOR_ANY:.*]] = builtin.unrealized_conversion_cast %[[TENS_ARG]] : !torch.vtensor<[4],f32> to !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[C32:.*]] = arith.constant 32 : index
// CHECK:      %[[C16:.*]] = arith.constant 16 : index
// CHECK:      %[[C128:.*]] = arith.constant 128 : index
// CHECK:      %[[C1:.*]] = arith.constant 1 : index
// CHECK:      %[[SHMEM:.*]] = arith.constant 16384 : i32
// CHECK:      %[[HANDLE_I64:.*]] = llvm.extractvalue %[[TENSOR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[HANDLE:.*]] = llvm.inttoptr %[[HANDLE_I64]] : i64 to !llvm.ptr
// CHECK:      %[[DLTENSOR_PTR:.*]] = llvm.getelementptr %[[HANDLE]]{{\[}}24] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:      %[[DATA_GEP:.*]] = llvm.getelementptr %[[DLTENSOR_PTR]]{{\[}}0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, struct<(i32, i32)>, i32, struct<(i8, i8, i16)>, ptr, ptr, i64)>
// CHECK:      %[[DATA_PTR:.*]] = llvm.load %[[DATA_GEP]] : !llvm.ptr -> !llvm.ptr
// CHECK:      %[[SCALAR_PLD:.*]] = llvm.extractvalue %[[SCALAR_ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:      %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:      %[[DEV_IDX_SLOT:.*]] = llvm.alloca %[[ONE]] x i32 : (i64) -> !llvm.ptr
// CHECK:      llvm.call @aoti_torch_get_current_device_index(%[[DEV_IDX_SLOT]]) : (!llvm.ptr) -> i32
// CHECK:      %[[DEVICE_IDX:.*]] = llvm.load %[[DEV_IDX_SLOT]] : !llvm.ptr -> i32
// CHECK:      %[[STRM_SLOT:.*]] = llvm.alloca %{{.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:      llvm.call @aoti_torch_get_current_stream(%[[DEVICE_IDX]], %[[STRM_SLOT]]) : (i32, !llvm.ptr) -> i32
// CHECK:      %[[ASYNC_OBJ:.*]] = llvm.load %[[STRM_SLOT]] : !llvm.ptr -> !llvm.ptr
// CHECK:      gpu.launch_func <%[[ASYNC_OBJ]] : !llvm.ptr> @kernel::@entry
// CHECK:      blocks in (%[[C32]], %[[C16]], %[[C1]])
// CHECK:      threads in (%[[C128]], %[[C1]], %[[C1]])
// CHECK:      dynamic_shared_memory_size %[[SHMEM]]
// CHECK:      args(%[[DATA_PTR]] : !llvm.ptr, %[[SCALAR_PLD]] : i64, %{{.*}} : i64, %{{.*}} : i64)

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
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %shmem = arith.constant 16384 : i32

    torchext.trident_kernel_launch @kernel::@entry
      blocks in (%c32, %c16, %c1) threads in (%c128, %c1, %c1)
      dynamic_shared_memory_size %shmem
      args (%tensor, %scalar : !torch.vtensor<[4],f32>, !torch.int)
      : index
    func.return
  }
}

// -----
// Test cast: !torch.float -> f32 (extractvalue + bitcast + fptrunc)
// CHECK-LABEL: func.func @cast_float_to_f32
// CHECK-SAME:  %[[ARG:.*]]: !torch.float
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.float to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[F64:.*]] = llvm.bitcast %[[PLD]] : i64 to f64
// CHECK:       llvm.fptrunc %[[F64]] : f64 to f32
// CHECK-NOT:   torchext.cast
func.func @cast_float_to_f32(%arg0: !torch.float) -> f32 {
  %0 = torchext.cast %arg0 : !torch.float -> f32
  return %0 : f32
}

// -----
// Test cast: !torch.float -> f64 (extractvalue + bitcast, no fptrunc)
// CHECK-LABEL: func.func @cast_float_to_f64
// CHECK-SAME:  %[[ARG:.*]]: !torch.float
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.float to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.bitcast %[[PLD]] : i64 to f64
// CHECK-NOT:   llvm.fptrunc
// CHECK-NOT:   torchext.cast
// CHECK:       return %{{.*}} : f64
func.func @cast_float_to_f64(%arg0: !torch.float) -> f64 {
  %0 = torchext.cast %arg0 : !torch.float -> f64
  return %0 : f64
}

// -----
// Test cast: !torch.int -> i32 (extractvalue + trunc)
// CHECK-LABEL: func.func @cast_int_to_i32
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.trunc %[[PLD]] : i64 to i32
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_i32(%arg0: !torch.int) -> i32 {
  %0 = torchext.cast %arg0 : !torch.int -> i32
  return %0 : i32
}

// -----
// Test cast: !torch.int -> i64 (extractvalue, no trunc)
// CHECK-LABEL: func.func @cast_int_to_i64
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NOT:   llvm.trunc
// CHECK-NOT:   torchext.cast
// CHECK:       return %[[PLD]] : i64
func.func @cast_int_to_i64(%arg0: !torch.int) -> i64 {
  %0 = torchext.cast %arg0 : !torch.int -> i64
  return %0 : i64
}

// -----
// Test cast: !torch.int -> i1 (extractvalue + trunc)
// CHECK-LABEL: func.func @cast_int_to_i1
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.trunc %[[PLD]] : i64 to i1
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_i1(%arg0: !torch.int) -> i1 {
  %0 = torchext.cast %arg0 : !torch.int -> i1
  return %0 : i1
}

// -----
// Test cast: !torch.int -> i8 (extractvalue + trunc)
// CHECK-LABEL: func.func @cast_int_to_i8
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.trunc %[[PLD]] : i64 to i8
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_i8(%arg0: !torch.int) -> i8 {
  %0 = torchext.cast %arg0 : !torch.int -> i8
  return %0 : i8
}

// -----
// Test cast: !torch.int -> i16 (extractvalue + trunc)
// CHECK-LABEL: func.func @cast_int_to_i16
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.trunc %[[PLD]] : i64 to i16
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_i16(%arg0: !torch.int) -> i16 {
  %0 = torchext.cast %arg0 : !torch.int -> i16
  return %0 : i16
}

// -----
// Test cast: !torch.int -> si32 (trunc to signless, then cast)
// CHECK-LABEL: func.func @cast_int_to_si32
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[TRUNC:.*]] = llvm.trunc %[[PLD]] : i64 to i32
// CHECK:       builtin.unrealized_conversion_cast %[[TRUNC]] : i32 to si32
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_si32(%arg0: !torch.int) -> si32 {
  %0 = torchext.cast %arg0 : !torch.int -> si32
  return %0 : si32
}

// -----
// Test cast: !torch.int -> ui32 (trunc to signless, then cast)
// CHECK-LABEL: func.func @cast_int_to_ui32
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[TRUNC:.*]] = llvm.trunc %[[PLD]] : i64 to i32
// CHECK:       builtin.unrealized_conversion_cast %[[TRUNC]] : i32 to ui32
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_ui32(%arg0: !torch.int) -> ui32 {
  %0 = torchext.cast %arg0 : !torch.int -> ui32
  return %0 : ui32
}

// -----
// Test cast: !torch.int -> si64 (no trunc needed, just cast)
// CHECK-LABEL: func.func @cast_int_to_si64
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       builtin.unrealized_conversion_cast %[[PLD]] : i64 to si64
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_si64(%arg0: !torch.int) -> si64 {
  %0 = torchext.cast %arg0 : !torch.int -> si64
  return %0 : si64
}

// -----
// Test cast: !torch.int -> ui64 (no trunc needed, just cast)
// CHECK-LABEL: func.func @cast_int_to_ui64
// CHECK-SAME:  %[[ARG:.*]]: !torch.int
// CHECK:       %[[ANY:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:.*]] = llvm.extractvalue %[[ANY]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       builtin.unrealized_conversion_cast %[[PLD]] : i64 to ui64
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_ui64(%arg0: !torch.int) -> ui64 {
  %0 = torchext.cast %arg0 : !torch.int -> ui64
  return %0 : ui64
}
