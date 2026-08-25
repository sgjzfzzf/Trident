//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

func.func @transpose(%arg0: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[3,2],f32> {
  // CHECK-LABEL: func.func @transpose(%arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
  // CHECK: %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.t"
  // CHECK: %[[CALL:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0)
  // CHECK-SAME: -> !tvm_ffi.tensor
  // CHECK: tvm_ffi.ObjectIncRef %[[CALL]]
  // CHECK: tvm_ffi.ObjectDecRef %[[CALL]]
  // CHECK: return %[[CALL]] : !tvm_ffi.tensor
  %0 = torch.aten.t %arg0
      : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  return %0 : !torch.vtensor<[3,2],f32>
}

// A value-semantic Torch clone may materialize a new memory layout before a
// raw-pointer kernel launch. It must reach the FFI lowering before Torch's
// folder, which ignores memory_format, can replace it with its input.
// CHECK-LABEL: func.func @clone_contiguous(
// CHECK: %[[FORMAT:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.clone"
// CHECK: %[[CLONE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %[[FORMAT]])
// CHECK-SAME: -> !tvm_ffi.tensor
// CHECK: return %[[CLONE]] : !tvm_ffi.tensor
func.func @clone_contiguous(%arg0: !torch.vtensor<[32,2],f32>)
    -> !torch.vtensor<[32,2],f32> {
  %memory_format = torch.constant.int 0
  %0 = torch.aten.clone %arg0, %memory_format
      : !torch.vtensor<[32,2],f32>, !torch.int
      -> !torch.vtensor<[32,2],f32>
  return %0 : !torch.vtensor<[32,2],f32>
}

// Non-ATen opaque operators are already legal for this conversion and must
// remain untouched.  ConvertAtenCall only handles generalized ATen names.
// CHECK-LABEL: func.func @opaque_operator() {
// CHECK-NEXT: torch.operator "torch.foo"() : () -> ()
// CHECK-NEXT: return
func.func @opaque_operator() {
  torch.operator "torch.foo"() : () -> ()
  return
}

// Constants are converted to their semantic TVM FFI scalar types, rather than
// being lowered through the LLVM representation used by the final ABI pass.
// CHECK-LABEL: func.func @constants(
// CHECK-SAME: -> !tvm_ffi.array {
// CHECK: %[[I0:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 7 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[I1:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 9 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[ARRAY:[a-zA-Z0-9_]+]] = "tvm_ffi.array.create"(%[[I0]], %[[I1]])
// CHECK: tvm_ffi.ObjectIncRef %[[ARRAY]] : !tvm_ffi.array
// CHECK: tvm_ffi.ObjectDecRef %[[ARRAY]] : !tvm_ffi.array
// CHECK: return %[[ARRAY]] : !tvm_ffi.array
func.func @constants() -> !torch.list<int> {
  %i = torch.constant.int 7
  %j = torch.constant.int 9
  %array = torch.prim.ListConstruct %i, %j
      : (!torch.int, !torch.int) -> !torch.list<int>
  return %array : !torch.list<int>
}

// CHECK-LABEL: func.func @constant_float() -> !tvm_ffi.float {
// CHECK: %[[FLOAT:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 2.500000e+00 : f64}> : () -> !tvm_ffi.float
// CHECK: return %[[FLOAT]] : !tvm_ffi.float
func.func @constant_float() -> !torch.float {
  %0 = torch.constant.float 2.5
  return %0 : !torch.float
}

// CHECK-LABEL: func.func @constant_bool() -> !tvm_ffi.bool {
// CHECK: %[[BOOL:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = true}> : () -> !tvm_ffi.bool
// CHECK: return %[[BOOL]] : !tvm_ffi.bool
func.func @constant_bool() -> !torch.bool {
  %0 = torch.constant.bool true
  return %0 : !torch.bool
}

// CHECK-LABEL: func.func @constant_none() -> !tvm_ffi.none {
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value}> : () -> !tvm_ffi.none
// CHECK: return %[[NONE]] : !tvm_ffi.none
func.func @constant_none() -> !torch.none {
  %0 = torch.constant.none
  return %0 : !torch.none
}

// TorchExt dtype values are already represented by TVM FFI dtypes when the
// Torch-to-TVMFFI conversion reaches a function boundary.
// CHECK-LABEL: func.func @dtype_identity(
// CHECK-SAME: %[[DTYPE:.*]]: !tvm_ffi.dtype) -> !tvm_ffi.dtype {
// CHECK: return %[[DTYPE]] : !tvm_ffi.dtype
func.func @dtype_identity(%arg0: !torchext.dtype) -> !torchext.dtype {
  return %arg0 : !torchext.dtype
}

// List and tuple containers share the TVM FFI array representation.
// CHECK-LABEL: func.func @container_construct(
// CHECK-SAME: -> !tvm_ffi.array {
// CHECK: "tvm_ffi.array.create"
func.func @container_construct(%arg0: !torch.int, %arg1: !torch.int)
    -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0, %arg1
      : (!torch.int, !torch.int) -> !torch.list<int>
  return %0 : !torch.list<int>
}

// Multiple ATen results are represented by an FFI array and extracted in
// result order.  This is the semantic counterpart of the single-result ABI
// packing performed by the later DecomposeTVMFFI/TVMFFIToLLVM passes.
// CHECK-LABEL: func.func @multi_result(
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: %[[KEEPDIM:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = false}> : () -> !tvm_ffi.bool
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.max.dim"
// CHECK: %[[PACKED:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%arg0, %[[DIM]], %[[KEEPDIM]]) : (!tvm_ffi.tensor, !tvm_ffi.int, !tvm_ffi.bool) -> !tvm_ffi.array
// CHECK: %[[IDX0:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 0 : i64}> : () -> !tvm_ffi.int
// CHECK: tvm_ffi.array.get_item %[[PACKED]][%[[IDX0]]] as !tvm_ffi.tensor
// CHECK: %[[IDX1:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value = 1 : i64}> : () -> !tvm_ffi.int
// CHECK: tvm_ffi.array.get_item %[[PACKED]][%[[IDX1]]] as !tvm_ffi.tensor
func.func @multi_result(%arg0: !torch.vtensor<[4],f32>)
    -> (!torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>) {
  %dim = torch.constant.int 0
  %keepdim = torch.constant.bool false
  %0, %1 = torch.aten.max.dim %arg0, %dim, %keepdim
      : !torch.vtensor<[4],f32>, !torch.int, !torch.bool
      -> !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>
  return %0, %1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4],si64>
}

// Each branch of torch.prim.If is processed with its own Region-local Set.
// The tensor produced by torch.aten.t is IncRef'd for the branch yield and
// DecRef'd for the branch-local ownership.
// CHECK-LABEL: func.func @nested_if
// CHECK: torch.prim.If
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: torch.prim.If.yield
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: torch.prim.If.yield
func.func @nested_if(%arg0: !torch.vtensor<[2,3],f32>, %cond: !torch.bool)
    -> !torch.vtensor<[3,2],f32> {
  %0 = torch.prim.If %cond -> (!torch.vtensor<[3,2],f32>) {
    %1 = torch.aten.t %arg0
        : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    torch.prim.If.yield %1 : !torch.vtensor<[3,2],f32>
  } else {
    %2 = torch.aten.t %arg0
        : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
    torch.prim.If.yield %2 : !torch.vtensor<[3,2],f32>
  }
  return %0 : !torch.vtensor<[3,2],f32>
}

// The TVMFFI function body is also a Region root, so its return is handled
// by the same terminator pattern.
// CHECK-LABEL: tvm_ffi.func @ffi_return
// CHECK-SAME: %arg0: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK: tvm_ffi.ObjectIncRef
// CHECK: tvm_ffi.ObjectDecRef
// CHECK: tvm_ffi.return
tvm_ffi.func @ffi_return(%arg0: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[3,2],f32> {
  %0 = torch.aten.t %arg0
      : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[3,2],f32>
  tvm_ffi.return %0 : !torch.vtensor<[3,2],f32>
}

// Container parameters are represented as one TVM FFI array at the ABI
// boundary. The conversion updates both the function signature and return.
// CHECK-LABEL: tvm_ffi.func @container_input(
// CHECK-SAME: %arg0: !tvm_ffi.array) -> !tvm_ffi.array {
// CHECK: tvm_ffi.return %arg0 : !tvm_ffi.array
tvm_ffi.func @container_input(
    %arg0: !torch.list<vtensor<[4],f32>>)
    -> !torch.list<vtensor<[4],f32>> {
  tvm_ffi.return %arg0 : !torch.list<vtensor<[4],f32>>
}

// Existing semantic inspection operations keep their operation and result
// types while all Torch operands are converted by the common conversion
// adaptor. Every inspection result remains live through the returned guard.
// CHECK-LABEL: tvm_ffi.func @guard_operand_conversion(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor,
// CHECK-SAME: %[[ARRAY:[a-zA-Z0-9_]+]]: !tvm_ffi.array,
// CHECK-SAME: %[[LHS:[a-zA-Z0-9_]+]]: !tvm_ffi.bool,
// CHECK-SAME: %[[RHS:[a-zA-Z0-9_]+]]: !tvm_ffi.bool) {
// CHECK-NOT: !torch
// CHECK: %[[EXPECTED_DEVICE_INDEX:[a-zA-Z0-9_]+]] = arith.constant 0 : i32
// CHECK: %[[EXPECTED_DEVICE_TYPE:[a-zA-Z0-9_]+]] = arith.constant 2 : i32
// CHECK: %[[EXPECTED_LANES:[a-zA-Z0-9_]+]] = arith.constant 1 : i16
// CHECK: %[[EXPECTED_BITS:[a-zA-Z0-9_]+]] = arith.constant 32 : i8
// CHECK: %[[EXPECTED_CODE:[a-zA-Z0-9_]+]] = arith.constant 2 : i8
// CHECK: %[[EXPECTED_STRIDE:[a-zA-Z0-9_]+]] = arith.constant 3 : i64
// CHECK: %[[EXPECTED_DIM_SIZE_LENGTH:[a-zA-Z0-9_]+]] = arith.constant 2 : i64
// CHECK: %[[INDEX_AND_EXPECTED_OFFSET:[a-zA-Z0-9_]+]] = arith.constant 0 : i64
// CHECK: %[[DIM:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.dim %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[SIZE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.size %[[TENSOR]][%[[INDEX_AND_EXPECTED_OFFSET]]] : !tvm_ffi.tensor
// CHECK: %[[STRIDE:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.stride %[[TENSOR]][%[[INDEX_AND_EXPECTED_OFFSET]]] : !tvm_ffi.tensor
// CHECK: %[[OFFSET:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.storage_offset %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[DTYPE_CODE:[a-zA-Z0-9_]+]], %[[DTYPE_BITS:[a-zA-Z0-9_]+]], %[[DTYPE_LANES:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.dtype %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[DEVICE_TYPE:[a-zA-Z0-9_]+]], %[[DEVICE_INDEX:[a-zA-Z0-9_]+]] = tvm_ffi.tensor.device %[[TENSOR]] : !tvm_ffi.tensor
// CHECK: %[[LENGTH:[a-zA-Z0-9_]+]] = tvm_ffi.array.length %[[ARRAY]] : !tvm_ffi.array
// CHECK: %[[VALUES_EQUAL:[a-zA-Z0-9_]+]] = tvm_ffi.eq %[[LHS]], %[[RHS]] : !tvm_ffi.bool, !tvm_ffi.bool
// CHECK: %[[DIM_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DIM]], %[[EXPECTED_DIM_SIZE_LENGTH]] : i64
// CHECK: %[[SIZE_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[SIZE]], %[[EXPECTED_DIM_SIZE_LENGTH]] : i64
// CHECK: %[[STRIDE_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[STRIDE]], %[[EXPECTED_STRIDE]] : i64
// CHECK: %[[OFFSET_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[OFFSET]], %[[INDEX_AND_EXPECTED_OFFSET]] : i64
// CHECK: %[[CODE_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DTYPE_CODE]], %[[EXPECTED_CODE]] : i8
// CHECK: %[[BITS_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DTYPE_BITS]], %[[EXPECTED_BITS]] : i8
// CHECK: %[[LANES_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DTYPE_LANES]], %[[EXPECTED_LANES]] : i16
// CHECK: %[[DEVICE_TYPE_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DEVICE_TYPE]], %[[EXPECTED_DEVICE_TYPE]] : i32
// CHECK: %[[DEVICE_INDEX_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[DEVICE_INDEX]], %[[EXPECTED_DEVICE_INDEX]] : i32
// CHECK: %[[LENGTH_OK:[a-zA-Z0-9_]+]] = arith.cmpi eq, %[[LENGTH]], %[[EXPECTED_DIM_SIZE_LENGTH]] : i64
// CHECK: %[[SHAPE_OK:[a-zA-Z0-9_]+]] = arith.andi %[[DIM_OK]], %[[SIZE_OK]] : i1
// CHECK: %[[LAYOUT_OK:[a-zA-Z0-9_]+]] = arith.andi %[[STRIDE_OK]], %[[OFFSET_OK]] : i1
// CHECK: %[[DTYPE_HEAD_OK:[a-zA-Z0-9_]+]] = arith.andi %[[CODE_OK]], %[[BITS_OK]] : i1
// CHECK: %[[DTYPE_OK:[a-zA-Z0-9_]+]] = arith.andi %[[DTYPE_HEAD_OK]], %[[LANES_OK]] : i1
// CHECK: %[[DEVICE_OK:[a-zA-Z0-9_]+]] = arith.andi %[[DEVICE_TYPE_OK]], %[[DEVICE_INDEX_OK]] : i1
// CHECK: %[[METADATA_HEAD_OK:[a-zA-Z0-9_]+]] = arith.andi %[[SHAPE_OK]], %[[LAYOUT_OK]] : i1
// CHECK: %[[METADATA_TAIL_OK:[a-zA-Z0-9_]+]] = arith.andi %[[DTYPE_OK]], %[[DEVICE_OK]] : i1
// CHECK: %[[METADATA_OK:[a-zA-Z0-9_]+]] = arith.andi %[[METADATA_HEAD_OK]], %[[METADATA_TAIL_OK]] : i1
// CHECK: %[[CONTAINER_OK:[a-zA-Z0-9_]+]] = arith.andi %[[LENGTH_OK]], %[[VALUES_EQUAL]] : i1
// CHECK: %[[GUARD_OK:[a-zA-Z0-9_]+]] = arith.andi %[[METADATA_OK]], %[[CONTAINER_OK]] : i1
// CHECK: %[[NONE:[a-zA-Z0-9_]+]] = "tvm_ffi.constant"() <{value}> : () -> !tvm_ffi.none
// CHECK: %[[SUCCESS:[a-zA-Z0-9_]+]] = tvm_ffi.cast %[[NONE]] : !tvm_ffi.none -> !tvm_ffi.any
// CHECK: %[[EXCEPTION:[a-zA-Z0-9_]+]] = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
// CHECK: %[[ERROR:[a-zA-Z0-9_]+]] = tvm_ffi.cast %[[EXCEPTION]] : !tvm_ffi.exception -> !tvm_ffi.any
// CHECK: %[[RESULT:[a-zA-Z0-9_]+]] = arith.select %[[GUARD_OK]], %[[SUCCESS]], %[[ERROR]] : !tvm_ffi.any
// CHECK: tvm_ffi.return %[[RESULT]] : !tvm_ffi.any
tvm_ffi.func @guard_operand_conversion(
    %tensor: !torch.vtensor<[2,3],f32>,
    %array: !torch.list<int>,
    %lhs: !torch.bool,
    %rhs: !torch.bool) {
  %index = arith.constant 0 : i64
  %dim = tvm_ffi.tensor.dim %tensor : !torch.vtensor<[2,3],f32>
  %size = tvm_ffi.tensor.size %tensor[%index]
      : !torch.vtensor<[2,3],f32>
  %stride = tvm_ffi.tensor.stride %tensor[%index]
      : !torch.vtensor<[2,3],f32>
  %offset = tvm_ffi.tensor.storage_offset %tensor
      : !torch.vtensor<[2,3],f32>
  %dtype_code, %dtype_bits, %dtype_lanes = tvm_ffi.tensor.dtype %tensor
      : !torch.vtensor<[2,3],f32>
  %device_type, %device_index = tvm_ffi.tensor.device %tensor
      : !torch.vtensor<[2,3],f32>
  %length = tvm_ffi.array.length %array : !torch.list<int>
  %values_equal = tvm_ffi.eq %lhs, %rhs : !torch.bool, !torch.bool

  %expected_dim = arith.constant 2 : i64
  %expected_size = arith.constant 2 : i64
  %expected_stride = arith.constant 3 : i64
  %expected_offset = arith.constant 0 : i64
  %expected_code = arith.constant 2 : i8
  %expected_bits = arith.constant 32 : i8
  %expected_lanes = arith.constant 1 : i16
  %expected_device_type = arith.constant 2 : i32
  %expected_device_index = arith.constant 0 : i32
  %expected_length = arith.constant 2 : i64
  %dim_ok = arith.cmpi eq, %dim, %expected_dim : i64
  %size_ok = arith.cmpi eq, %size, %expected_size : i64
  %stride_ok = arith.cmpi eq, %stride, %expected_stride : i64
  %offset_ok = arith.cmpi eq, %offset, %expected_offset : i64
  %code_ok = arith.cmpi eq, %dtype_code, %expected_code : i8
  %bits_ok = arith.cmpi eq, %dtype_bits, %expected_bits : i8
  %lanes_ok = arith.cmpi eq, %dtype_lanes, %expected_lanes : i16
  %device_type_ok = arith.cmpi eq, %device_type, %expected_device_type : i32
  %device_index_ok = arith.cmpi eq, %device_index, %expected_device_index : i32
  %length_ok = arith.cmpi eq, %length, %expected_length : i64
  %tensor_shape_ok = arith.andi %dim_ok, %size_ok : i1
  %tensor_layout_ok = arith.andi %stride_ok, %offset_ok : i1
  %tensor_dtype_head_ok = arith.andi %code_ok, %bits_ok : i1
  %tensor_dtype_ok = arith.andi %tensor_dtype_head_ok, %lanes_ok : i1
  %tensor_device_ok = arith.andi %device_type_ok, %device_index_ok : i1
  %tensor_metadata_head_ok = arith.andi %tensor_shape_ok, %tensor_layout_ok : i1
  %tensor_metadata_tail_ok = arith.andi %tensor_dtype_ok, %tensor_device_ok : i1
  %tensor_metadata_ok = arith.andi %tensor_metadata_head_ok,
      %tensor_metadata_tail_ok : i1
  %container_ok = arith.andi %length_ok, %values_equal : i1
  %guard_ok = arith.andi %tensor_metadata_ok, %container_ok : i1

  %none = "tvm_ffi.constant"() <{value}> : () -> !tvm_ffi.none
  %success = tvm_ffi.cast %none : !tvm_ffi.none -> !tvm_ffi.any
  %exception = tvm_ffi.exception "GuardMatch" : !tvm_ffi.exception
  %error = tvm_ffi.cast %exception : !tvm_ffi.exception -> !tvm_ffi.any
  %result = arith.select %guard_ok, %success, %error : !tvm_ffi.any
  tvm_ffi.return %result : !tvm_ffi.any
}

// The pass can run by itself because it declares the LLVM dialect used by
// tensor literal lowering. Dialect conversion materializes the semantic tensor
// boundary directly, without leaving a Torch-typed cast.
// CHECK-LABEL: func.func @literal_standalone() -> !tvm_ffi.tensor {
// CHECK-DAG: llvm.mlir.addressof @__trident_constant_trident.runtime.tensor_to_tvm_ffi_object_trident.runtime.tensor_to_tvm_ffi_object : !llvm.ptr
// CHECK-DAG: llvm.mlir.addressof @__trident_constant_trident.runtime.tvm_ffi_to_torch_type_trident.runtime.tvm_ffi_to_torch_type : !llvm.ptr
// CHECK-DAG: llvm.mlir.addressof @__trident_constant_trident.runtime.tvm_ffi_device_to_torch_device_type_trident.runtime.tvm_ffi_device_to_torch_device_type : !llvm.ptr
// CHECK-DAG: llvm.call @TVMFFIFunctionGetGlobal
// CHECK-DAG: llvm.call @TVMFFIFunctionGetGlobal
// CHECK-DAG: llvm.call @TVMFFIFunctionGetGlobal
// CHECK-DAG: llvm.call @TVMFFIFunctionCall
// CHECK-DAG: llvm.call @TVMFFIFunctionCall
// CHECK-DAG: llvm.call @TVMFFIFunctionCall
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: builtin.unrealized_conversion_cast {{.*}} : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.tensor
// CHECK: return {{.*}} : !tvm_ffi.tensor
func.func @literal_standalone() -> !torch.vtensor<[2,3],f32> {
  %literal = torch.vtensor.literal(
      dense<1.250000e+00> : tensor<2x3xf32>)
      : !torch.vtensor<[2,3],f32>
  return %literal : !torch.vtensor<[2,3],f32>
}
