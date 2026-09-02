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
// CHECK-NOT:   torchext.get
// CHECK:       return %[[BC]] : f64
func.func @cast_float_to_f64(%arg0: !torch.float) -> f64 {
  %0 = torchext.get %arg0 : !torch.float -> f64
  return %0 : f64
}

// -----
// Test get: !torch.int -> i64 (extractvalue)
// CHECK-LABEL: func.func @cast_int_to_i64
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK-NOT:   llvm.trunc
// CHECK-NOT:   torchext.get
// CHECK:       return %[[PLD]] : i64
func.func @cast_int_to_i64(%arg0: !torch.int) -> i64 {
  %0 = torchext.get %arg0 : !torch.int -> i64
  return %0 : i64
}
