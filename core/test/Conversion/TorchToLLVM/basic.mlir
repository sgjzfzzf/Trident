//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s --convert-torch-to-llvm | FileCheck %s
//
// Tests that the standalone ConvertTorchToLLVM pass lowers
// torch.constant.bool/int/float/none to TVMFFIAny structs, and lowers
// torch.prim.ListUnpack on !torch.list / !torch.tuple containers into
// ffi.ArrayGetItem runtime calls.
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
// Build slot pointers and call ffi.Array via TVMFFIFunctionGetGlobal.
// CHECK:           llvm.getelementptr %[[ARR]][0]
// CHECK:           llvm.getelementptr %[[ARR]][1]
// CHECK:           llvm.mlir.addressof @__trident_constant_ffi.Array_ffi.Array
// CHECK:           llvm.call @TVMFFIFunctionGetGlobal
// TVMFFIFunctionCall + TVMFFIObjectDecRef
// CHECK:           llvm.call @TVMFFIFunctionCall
// CHECK:           llvm.call @TVMFFIObjectDecRef
// Build TVMFFIAny(kTVMFFIArray=71) from the ffi.Array result.
// CHECK:           llvm.getelementptr {{%.*}}[0, 2]
// CHECK:           %[[VOBJ:.*]] = llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK:           llvm.mlir.undef : !llvm.struct<(i32, i32, i64)>
// CHECK:           llvm.mlir.constant(71 : i32) : i32
// CHECK:           llvm.insertvalue {{%.*}}, {{%.*}}[0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           llvm.insertvalue %[[VOBJ]], {{%.*}}[2] : !llvm.struct<(i32, i32, i64)>
// Wrap in unrealized_conversion_cast for the Torch return type.
// CHECK:           builtin.unrealized_conversion_cast {{%.*}} : !llvm.struct<(i32, i32, i64)> to !torch.list<int>
// The list result escapes through the return: the ref-counting pass inserts an
// IncRef (escape) followed by a DecRef (last use in this scope), which cancel
// out at runtime and hand the caller a reference-count-1 list.
// CHECK:           llvm.call @TVMFFIObjectIncRef
// CHECK:           llvm.call @TVMFFIObjectDecRef
// CHECK:           return {{%.*}} : !torch.list<int>
func.func @torch.prim.list_construct(%arg0: !torch.int, %arg1: !torch.int) -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  return %0 : !torch.list<int>
}

// The list container is destructured via ffi.ArrayGetItem runtime calls, one
// per result.  The container operand (a Torch-typed block argument) is
// materialized to TVMFFIAny; each extracted element is a *borrowed*
// reference, so no ref-counting is emitted.
// CHECK-LABEL:   func.func @unpack_list(
// CHECK-SAME:      %[[CONTAINER:.*]]: !torch.list<vtensor<[4],f32>>) -> !torch.vtensor<[4],f32> {
// CHECK:           [[CASTED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast %[[CONTAINER]] : !torch.list<vtensor<[4],f32>> to !llvm.struct<(i32, i32, i64)>
// The shared 2-slot args array is allocated once; the container fills slot[0].
// CHECK:           [[ARGS_SLOT:%[a-z0-9]+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i64)>
// CHECK:           [[SLOT0:%[a-z0-9]+]] = llvm.getelementptr [[ARGS_SLOT]][0]
// CHECK:           llvm.store [[CASTED]], [[SLOT0]]
// Each extraction resolves, calls, and releases ffi.ArrayGetItem.
// Index args are full TVMFFIAny (kTVMFFIInt) slots.
// CHECK:           [[IDX_SLOT:%[a-z0-9]+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i64)>
// CHECK:           llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(i32, i32, i64)>
// CHECK:           llvm.call @TVMFFIFunctionGetGlobal{{.*}} : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.call @TVMFFIFunctionCall{{.*}} : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK:           llvm.call @TVMFFIObjectDecRef{{.*}} : (!llvm.ptr) -> i32
// CHECK:           [[E0:%[a-z0-9]+]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:           [[C0:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[E0]] : !llvm.struct<(i32, i32, i64)> to !torch.vtensor<[4],f32>
// CHECK:           llvm.call @TVMFFIFunctionGetGlobal{{.*}} : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.call @TVMFFIFunctionCall{{.*}} : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK:           llvm.call @TVMFFIObjectDecRef{{.*}} : (!llvm.ptr) -> i32
// CHECK:           return [[C0]] : !torch.vtensor<[4],f32>
func.func @unpack_list(%arg0: !torch.list<vtensor<[4],f32>>) -> !torch.vtensor<[4],f32> {
  %0, %1 = torch.prim.ListUnpack %arg0 : !torch.list<vtensor<[4],f32>> -> !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>
  return %0 : !torch.vtensor<[4],f32>
}

// A tvm_ffi.func wrapper (as emitted by the Python frontend): the tuple
// parameter is typed !torch.tuple<...>, destructured in the body, and the
// flattened leaves are passed to the callee in order.
// CHECK-LABEL:   tvm_ffi.func @unpack_tuple(
// CHECK-SAME:      %[[CONTAINER:.*]]: !torch.tuple<vtensor<[4],f32>, vtensor<[4],f32>>) -> !torch.vtensor<[4],f32> {
// CHECK:           [[CASTED:%[a-z0-9]+]] = builtin.unrealized_conversion_cast %[[CONTAINER]] : !torch.tuple<vtensor<[4],f32>, vtensor<[4],f32>> to !llvm.struct<(i32, i32, i64)>
// Each extraction resolves, calls, and releases ffi.ArrayGetItem.
// CHECK:           llvm.call @TVMFFIFunctionGetGlobal{{.*}} : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.call @TVMFFIFunctionCall{{.*}} : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK:           llvm.call @TVMFFIObjectDecRef{{.*}} : (!llvm.ptr) -> i32
// CHECK:           [[E0:%[a-z0-9]+]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:           [[C0:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[E0]] : !llvm.struct<(i32, i32, i64)> to !torch.vtensor<[4],f32>
// CHECK:           llvm.call @TVMFFIFunctionGetGlobal{{.*}} : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           llvm.call @TVMFFIFunctionCall{{.*}} : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK:           llvm.call @TVMFFIObjectDecRef{{.*}} : (!llvm.ptr) -> i32
// CHECK:           [[E1:%[a-z0-9]+]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:           [[C1:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[E1]] : !llvm.struct<(i32, i32, i64)> to !torch.vtensor<[4],f32>
// CHECK:           {{%.*}} = func.call @callee{{.*}} : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
// CHECK:           tvm_ffi.return {{%.*}} : !torch.vtensor<[4],f32>
tvm_ffi.func @unpack_tuple(%arg0: !torch.tuple<vtensor<[4],f32>, vtensor<[4],f32>>) -> !torch.vtensor<[4],f32> {
  %0, %1 = torch.prim.ListUnpack %arg0 : !torch.tuple<vtensor<[4],f32>, vtensor<[4],f32>> -> !torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>
  %2 = func.call @callee(%0, %1) : (!torch.vtensor<[4],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32>
  tvm_ffi.return %2 : !torch.vtensor<[4],f32>
}

func.func @callee(%arg0: !torch.vtensor<[4],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> {
  return %arg0 : !torch.vtensor<[4],f32>
}

// Nested containers are unpacked recursively.
// CHECK-LABEL:   func.func @unpack_nested(
// CHECK-SAME:      %[[CONTAINER:.*]]: !torch.list<list<vtensor<[4],f32>>>) -> !torch.vtensor<[4],f32> {
// CHECK-DAG:       llvm.call @TVMFFIFunctionCall{{.*}} : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK-DAG:       llvm.call @TVMFFIFunctionCall{{.*}} : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK:           [[ELEM:%[a-z0-9]+]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:           [[CAST:%[a-z0-9]+]] = builtin.unrealized_conversion_cast [[ELEM]] : !llvm.struct<(i32, i32, i64)> to !torch.vtensor<[4],f32>
// CHECK:           return [[CAST]] : !torch.vtensor<[4],f32>
func.func @unpack_nested(%arg0: !torch.list<list<vtensor<[4],f32>>>) -> !torch.vtensor<[4],f32> {
  %0 = torch.prim.ListUnpack %arg0 : !torch.list<list<vtensor<[4],f32>>> -> !torch.list<vtensor<[4],f32>>
  %1 = torch.prim.ListUnpack %0 : !torch.list<vtensor<[4],f32>> -> !torch.vtensor<[4],f32>
  return %1 : !torch.vtensor<[4],f32>
}
