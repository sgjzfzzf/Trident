//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torchext-to-gpu --convert-torch-to-tvm-ffi --convert-tvm-ffi-to-func --convert-tvm-ffi-to-llvm -split-input-file | FileCheck %s

// Test get: !torch.float -> f64 (extractvalue + bitcast)
// CHECK-LABEL: func.func @cast_float_to_f64
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[BC:[a-zA-Z0-9_]+]] = llvm.bitcast %[[PLD]] : i64 to f64
// CHECK-NOT:   llvm.fptrunc
// CHECK-NOT:   torch_c.to_f64
// CHECK:       return %[[BC]] : f64
func.func @cast_float_to_f64(%arg0: !torch.float) -> f64 {
  %0 = torch_c.to_f64 %arg0
  return %0 : f64
}

// -----
// Test get: !torch.int -> i64 (extractvalue)
// CHECK-LABEL: func.func @cast_int_to_i64
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NOT:   llvm.trunc
// CHECK-NOT:   torch_c.to_i64
// CHECK:       return %[[PLD]] : i64
func.func @cast_int_to_i64(%arg0: !torch.int) -> i64 {
  %0 = torch_c.to_i64 %arg0
  return %0 : i64
}

// -----
// Test the kernel-argument construction shape: !torch.int -> i64 -> i32
// CHECK-LABEL: func.func @cast_int_to_i32
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[I32:[a-zA-Z0-9_]+]] = llvm.trunc %[[PLD]] : i64 to i32
// CHECK-NOT:   torch_c.to_i64
// CHECK-NOT:   tvm_ffi.get
// CHECK:       return %[[I32]] : i32
func.func @cast_int_to_i32(%arg0: !torch.int) -> i32 {
  %native = torch_c.to_i64 %arg0
  %0 = llvm.trunc %native : i64 to i32
  return %0 : i32
}
