//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torch-to-tvm-ffi --convert-tvm-ffi-to-func --convert-tvm-ffi-to-llvm --convert-torchext-to-gpu -split-input-file | FileCheck %s

// Test cast: !torch.float -> f32 (extractvalue + bitcast + fptrunc)
// CHECK-LABEL: func.func @cast_float_to_f32
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[F64:[a-zA-Z0-9_]+]] = llvm.bitcast %[[PLD]] : i64 to f64
// CHECK:       llvm.fptrunc %[[F64]] : f64 to f32
// CHECK-NOT:   torchext.cast
func.func @cast_float_to_f32(%arg0: !torch.float) -> f32 {
  %0 = torchext.cast %arg0 : !torch.float -> f32
  return %0 : f32
}

// -----
// Test cast: !torch.float -> f64 (extractvalue + bitcast, no fptrunc)
// CHECK-LABEL: func.func @cast_float_to_f64
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[BC:[a-zA-Z0-9_]+]] = llvm.bitcast %[[PLD]] : i64 to f64
// CHECK-NOT:   llvm.fptrunc
// CHECK-NOT:   torchext.cast
// CHECK:       return %[[BC]] : f64
func.func @cast_float_to_f64(%arg0: !torch.float) -> f64 {
  %0 = torchext.cast %arg0 : !torch.float -> f64
  return %0 : f64
}

// -----
// Test cast: !torch.int -> i32 (extractvalue + trunc)
// CHECK-LABEL: func.func @cast_int_to_i32
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.trunc %[[PLD]] : i64 to i32
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_i32(%arg0: !torch.int) -> i32 {
  %0 = torchext.cast %arg0 : !torch.int -> i32
  return %0 : i32
}

// -----
// Test cast: !torch.int -> i64 (extractvalue, no trunc)
// CHECK-LABEL: func.func @cast_int_to_i64
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
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
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.trunc %[[PLD]] : i64 to i1
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_i1(%arg0: !torch.int) -> i1 {
  %0 = torchext.cast %arg0 : !torch.int -> i1
  return %0 : i1
}

// -----
// Test cast: !torch.int -> i8 (extractvalue + trunc)
// CHECK-LABEL: func.func @cast_int_to_i8
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.trunc %[[PLD]] : i64 to i8
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_i8(%arg0: !torch.int) -> i8 {
  %0 = torchext.cast %arg0 : !torch.int -> i8
  return %0 : i8
}

// -----
// Test cast: !torch.int -> i16 (extractvalue + trunc)
// CHECK-LABEL: func.func @cast_int_to_i16
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       llvm.trunc %[[PLD]] : i64 to i16
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_i16(%arg0: !torch.int) -> i16 {
  %0 = torchext.cast %arg0 : !torch.int -> i16
  return %0 : i16
}

// -----
// Test cast: !torch.int -> si32 (trunc to signless, then cast)
// CHECK-LABEL: func.func @cast_int_to_si32
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[TRUNC:[a-zA-Z0-9_]+]] = llvm.trunc %[[PLD]] : i64 to i32
// CHECK:       builtin.unrealized_conversion_cast %[[TRUNC]] : i32 to si32
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_si32(%arg0: !torch.int) -> si32 {
  %0 = torchext.cast %arg0 : !torch.int -> si32
  return %0 : si32
}

// -----
// Test cast: !torch.int -> ui32 (trunc to signless, then cast)
// CHECK-LABEL: func.func @cast_int_to_ui32
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[TRUNC:[a-zA-Z0-9_]+]] = llvm.trunc %[[PLD]] : i64 to i32
// CHECK:       builtin.unrealized_conversion_cast %[[TRUNC]] : i32 to ui32
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_ui32(%arg0: !torch.int) -> ui32 {
  %0 = torchext.cast %arg0 : !torch.int -> ui32
  return %0 : ui32
}

// -----
// Test cast: !torch.int -> si64 (no trunc needed, just cast)
// CHECK-LABEL: func.func @cast_int_to_si64
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       builtin.unrealized_conversion_cast %[[PLD]] : i64 to si64
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_si64(%arg0: !torch.int) -> si64 {
  %0 = torchext.cast %arg0 : !torch.int -> si64
  return %0 : si64
}

// -----
// Test cast: !torch.int -> ui64 (no trunc needed, just cast)
// CHECK-LABEL: func.func @cast_int_to_ui64
// CHECK-SAME:  %[[ARG:[a-zA-Z0-9_]+]]: !llvm.struct<(i32, i32, i64)>
// CHECK:       %[[PLD:[a-zA-Z0-9_]+]] = llvm.extractvalue %[[ARG]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:       builtin.unrealized_conversion_cast %[[PLD]] : i64 to ui64
// CHECK-NOT:   torchext.cast
func.func @cast_int_to_ui64(%arg0: !torch.int) -> ui64 {
  %0 = torchext.cast %arg0 : !torch.int -> ui64
  return %0 : ui64
}
