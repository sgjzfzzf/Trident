//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torch-to-llvm | FileCheck %s
//
// Tests that the standalone ConvertTorchToLLVM pass lowers
// torch.constant.bool/int/float/none to TVMFFIAny structs.
// Function signatures remain unchanged (type conversion is handled by the
// separate func-backend-type-conversion pass).

// CHECK-LABEL:   func.func @torch.constant.bool() -> !torch.bool {
// CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[IDX:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[WITH_IDX:.*]] = llvm.insertvalue %[[IDX]], %[[UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[PAYLOAD:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[WITH_PLD:.*]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_IDX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[WITH_PLD]] : !llvm.struct<(i32, i32, i64)> to !torch.bool
// CHECK-NEXT:      return %[[C]] : !torch.bool
func.func @torch.constant.bool() -> !torch.bool {
  %true = torch.constant.bool true
  return %true : !torch.bool
}

// CHECK-LABEL:   func.func @torch.constant.int() -> !torch.int {
// CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[IDX:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[WITH_IDX:.*]] = llvm.insertvalue %[[IDX]], %[[UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[PAYLOAD:.*]] = llvm.mlir.constant(42 : i64) : i64
// CHECK:           %[[WITH_PLD:.*]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_IDX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[WITH_PLD]] : !llvm.struct<(i32, i32, i64)> to !torch.int
// CHECK-NEXT:      return %[[C]] : !torch.int
func.func @torch.constant.int() -> !torch.int {
  %int = torch.constant.int 42
  return %int : !torch.int
}

// CHECK-LABEL:   func.func @torch.constant.float() -> !torch.float {
// CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[IDX:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[WITH_IDX:.*]] = llvm.insertvalue %[[IDX]], %[[UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[FVAL:.*]] = llvm.mlir.constant(3.140000e+00 : f64) : f64
// CHECK:           %[[BC:.*]] = llvm.bitcast %[[FVAL]] : f64 to i64
// CHECK:           %[[WITH_PLD:.*]] = llvm.insertvalue %[[BC]], %[[WITH_IDX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[WITH_PLD]] : !llvm.struct<(i32, i32, i64)> to !torch.float
// CHECK-NEXT:      return %[[C]] : !torch.float
func.func @torch.constant.float() -> !torch.float {
  %float = torch.constant.float 3.14
  return %float : !torch.float
}

// CHECK-LABEL:   func.func @torch.constant.none() -> !torch.none {
// CHECK:           %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[IDX:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[WITH_IDX:.*]] = llvm.insertvalue %[[IDX]], %[[UNDEF]][0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[PAYLOAD:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[WITH_PLD:.*]] = llvm.insertvalue %[[PAYLOAD]], %[[WITH_IDX]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           %[[C:.*]] = builtin.unrealized_conversion_cast %[[WITH_PLD]] : !llvm.struct<(i32, i32, i64)> to !torch.none
// CHECK-NEXT:      return %[[C]] : !torch.none
func.func @torch.constant.none() -> !torch.none {
  %none = torch.constant.none
  return %none : !torch.none
}

// CHECK-LABEL:   func.func @torch.prim.list_construct(
// CHECK-SAME:      %[[A:.*]]: !torch.int, %[[B:.*]]: !torch.int) -> !torch.list<int> {
// The adapted values come through unrealized_conversion_cast from Torch types.
// CHECK-DAG:      %[[A_ADAPT:.*]] = builtin.unrealized_conversion_cast %[[A]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// CHECK-DAG:      %[[B_ADAPT:.*]] = builtin.unrealized_conversion_cast %[[B]] : !torch.int to !llvm.struct<(i32, i32, i64)>
// Allocate array of 2 TVMFFIAny structs.
// CHECK:           %[[N:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:      %[[ARR:.*]] = llvm.alloca %[[N]] x !llvm.struct<(i32, i32, i64)>
// Extract payloads and fill slots.
// CHECK:           llvm.extractvalue %[[A_ADAPT]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           llvm.getelementptr %[[ARR]][0]
// CHECK:           llvm.getelementptr {{%.*}}[0, 0]
// CHECK:           llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:           llvm.getelementptr {{%.*}}[0, 1]
// CHECK:           llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:           llvm.getelementptr {{%.*}}[0, 2]
// CHECK:           llvm.store {{%.*}}, {{%.*}} : i64, !llvm.ptr
// CHECK:           llvm.extractvalue %[[B_ADAPT]][2] : !llvm.struct<(i32, i32, i64)>
// CHECK:           llvm.getelementptr %[[ARR]][1]
// CHECK:           llvm.getelementptr {{%.*}}[0, 0]
// CHECK:           llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:           llvm.getelementptr {{%.*}}[0, 1]
// CHECK:           llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:           llvm.getelementptr {{%.*}}[0, 2]
// CHECK:           llvm.store {{%.*}}, {{%.*}} : i64, !llvm.ptr
// Build slot pointers and call ffi.Array through its cached function handle.
// CHECK:           llvm.getelementptr %[[ARR]][0]
// CHECK:           llvm.getelementptr %[[ARR]][1]
// CHECK:           %[[HANDLE_ADDR:.*]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_ffi.Array
// CHECK-NEXT:      %[[HANDLE:.*]] = llvm.load %[[HANDLE_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK-NOT:       llvm.call @TVMFFIFunctionGetGlobal
// CHECK-NOT:       llvm.call @TVMFFIObjectDecRef
// CHECK:           llvm.call @TVMFFIFunctionCall(%[[HANDLE]],
// Build TVMFFIAny(kTVMFFIArray=71) from the ffi.Array result.
// CHECK:           llvm.getelementptr {{%.*}}[0, 2]
// CHECK:           %[[VOBJ:.*]] = llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK:           llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           llvm.mlir.constant(71 : i32) : i32
// CHECK:           llvm.insertvalue {{%.*}}, {{%.*}}[0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           llvm.insertvalue %[[VOBJ]], {{%.*}}[2] : !llvm.struct<(i32, i32, i64)>
// Wrap in unrealized_conversion_cast for the Torch return type.
// CHECK:           builtin.unrealized_conversion_cast {{%.*}} : !llvm.struct<(i32, i32, i64)> to !torch.list<int>
// CHECK-NEXT:      return {{%.*}} : !torch.list<int>
func.func @torch.prim.list_construct(%arg0: !torch.int, %arg1: !torch.int) -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  return %0 : !torch.list<int>
}

// CHECK-LABEL: llvm.func internal @__trident_tvm_ffi_ctor_ffi.Array() {
// CHECK:         llvm.mlir.addressof @__trident_constant_ffi.Array_ffi.Array
// CHECK:         %[[GET_GLOBAL:.*]] = llvm.call @TVMFFIFunctionGetGlobal
// CHECK:         %[[CTOR_HANDLE:.*]] = llvm.load
// CHECK:         %[[GLOBAL_ADDR:.*]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_ffi.Array
// CHECK:         llvm.store %[[CTOR_HANDLE]], %[[GLOBAL_ADDR]] : !llvm.ptr, !llvm.ptr
// CHECK:         llvm.return

// CHECK-LABEL: llvm.func internal @__trident_tvm_ffi_dtor_ffi.Array() {
// CHECK:         %[[DTOR_ADDR:.*]] = llvm.mlir.addressof @__trident_tvm_ffi_handle_ffi.Array
// CHECK:         %[[DTOR_HANDLE:.*]] = llvm.load %[[DTOR_ADDR]] : !llvm.ptr -> !llvm.ptr
// CHECK:         llvm.call @TVMFFIObjectDecRef(%[[DTOR_HANDLE]])
// CHECK:         llvm.return

// CHECK: llvm.mlir.global_ctors ctors = [@__trident_tvm_ffi_ctor_ffi.Array], priorities = [65535 : i32], data = [#llvm.zero]
// CHECK: llvm.mlir.global_dtors dtors = [@__trident_tvm_ffi_dtor_ffi.Array], priorities = [65535 : i32], data = [#llvm.zero]
