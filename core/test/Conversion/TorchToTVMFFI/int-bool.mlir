//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s
// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi -convert-tvm-ffi-to-llvm | FileCheck %s --check-prefix=LLVM

// CHECK-LABEL: func.func @int_bool(
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = tvm_ffi.get %arg0 : !tvm_ffi.bool -> i1
// CHECK: %[[INT:[a-zA-Z0-9_]+]] = arith.extui %[[BOOL]] : i1 to i64
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.to %[[INT]] : i64 -> !tvm_ffi.int
// CHECK: return %[[RESULT]] : !tvm_ffi.int

// LLVM-LABEL: func.func @int_bool(
// LLVM-NOT: builtin.unrealized_conversion_cast
// LLVM-NOT: ConvertUnrealizedIntCastOp
// LLVM-NOT: tvm_ffi.to
// LLVM-NOT: torch_c.from_i64
// LLVM: %[[PAYLOAD:[a-zA-Z0-9_]+]] = llvm.extractvalue %arg0[2]
// LLVM: %[[BOOL:[a-zA-Z0-9_]+]] = llvm.trunc %[[PAYLOAD]] : i64 to i1
// LLVM: %[[INT:[a-zA-Z0-9_]+]] = arith.extui %[[BOOL]] : i1 to i64
// LLVM: %[[TYPE_INDEX:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32)
// LLVM: llvm.insertvalue %[[TYPE_INDEX]], {{.*}}[0]
// LLVM: llvm.insertvalue %[[INT]], {{.*}}[2]
func.func @int_bool(%arg0: !torch.bool) -> !torch.int {
  %0 = torch.aten.Int.bool %arg0 : !torch.bool -> !torch.int
  return %0 : !torch.int
}
