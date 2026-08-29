//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

// TorchExt dtype values are already represented by TVM FFI dtypes when the
// Torch-to-TVMFFI conversion reaches a function boundary.
// CHECK-LABEL: func.func @dtype_identity(
// CHECK-SAME: %[[DTYPE:[a-zA-Z0-9_]+]]: !tvm_ffi.dtype) -> !tvm_ffi.dtype {
// CHECK: return %[[DTYPE]] : !tvm_ffi.dtype
func.func @dtype_identity(%arg0: !torchext.dtype) -> !torchext.dtype {
  return %arg0 : !torchext.dtype
}

// CHECK-LABEL: func.func @dtype_to_torch_type(
// CHECK-SAME: %[[DTYPE:[a-zA-Z0-9_]+]]: !tvm_ffi.dtype) -> !tvm_ffi.int {
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.runtime.tvm_ffi_to_torch_type"
// CHECK: %[[TYPE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%[[DTYPE]])
// CHECK-SAME: (!tvm_ffi.dtype) -> !tvm_ffi.int
// CHECK: return %[[TYPE]] : !tvm_ffi.int
func.func @dtype_to_torch_type(%dtype: !torchext.dtype) -> !torch.int {
  %type = torchext.convert %dtype : !torchext.dtype -> !torch.int
  return %type : !torch.int
}

// Torch Any is a genuinely dynamic ABI value, not an alias for an array.
// CHECK-LABEL: func.func @any_identity(
// CHECK-SAME: %[[ANY:[a-zA-Z0-9_]+]]: !tvm_ffi.any) -> !tvm_ffi.any {
// CHECK: return %[[ANY]] : !tvm_ffi.any
func.func @any_identity(%arg0: !torch.any) -> !torch.any {
  return %arg0 : !torch.any
}

// CHECK-LABEL: func.func @device_identity(
// CHECK-SAME: %[[DEVICE:[a-zA-Z0-9_]+]]: !tvm_ffi.device) -> !tvm_ffi.device {
// CHECK: return %[[DEVICE]] : !tvm_ffi.device
func.func @device_identity(%arg0: !torch.Device) -> !torch.Device {
  return %arg0 : !torch.Device
}

// CHECK-LABEL: func.func @bool_identity(
// CHECK-SAME: %[[BOOL:[a-zA-Z0-9_]+]]: !tvm_ffi.bool) -> !tvm_ffi.bool {
// CHECK: return %[[BOOL]] : !tvm_ffi.bool
func.func @bool_identity(%arg0: !torch.bool) -> !torch.bool {
  return %arg0 : !torch.bool
}

// CHECK-LABEL: func.func @float_identity(
// CHECK-SAME: %[[FLOAT:[a-zA-Z0-9_]+]]: !tvm_ffi.float) -> !tvm_ffi.float {
// CHECK: return %[[FLOAT]] : !tvm_ffi.float
func.func @float_identity(%arg0: !torch.float) -> !torch.float {
  return %arg0 : !torch.float
}

// CHECK-LABEL: func.func @int_identity(
// CHECK-SAME: %[[INT:[a-zA-Z0-9_]+]]: !tvm_ffi.int) -> !tvm_ffi.int {
// CHECK: return %[[INT]] : !tvm_ffi.int
func.func @int_identity(%arg0: !torch.int) -> !torch.int {
  return %arg0 : !torch.int
}

// CHECK-LABEL: func.func @none_identity(
// CHECK-SAME: %[[NONE:[a-zA-Z0-9_]+]]: !tvm_ffi.none) -> !tvm_ffi.none {
// CHECK: return %[[NONE]] : !tvm_ffi.none
func.func @none_identity(%arg0: !torch.none) -> !torch.none {
  return %arg0 : !torch.none
}

// CHECK-LABEL: func.func @list_identity(
// CHECK-SAME: %[[LIST:[a-zA-Z0-9_]+]]: !tvm_ffi.array) -> !tvm_ffi.array {
// CHECK: return %[[LIST]] : !tvm_ffi.array
func.func @list_identity(%arg0: !torch.list<int>) -> !torch.list<int> {
  return %arg0 : !torch.list<int>
}

// CHECK-LABEL: func.func @tuple_identity(
// CHECK-SAME: %[[TUPLE:[a-zA-Z0-9_]+]]: !tvm_ffi.array) -> !tvm_ffi.array {
// CHECK: return %[[TUPLE]] : !tvm_ffi.array
func.func @tuple_identity(%arg0: !torch.tuple<int, int>)
    -> !torch.tuple<int, int> {
  return %arg0 : !torch.tuple<int, int>
}

// CHECK-LABEL: func.func @tensor_identity(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK: return %[[TENSOR]] : !tvm_ffi.tensor
func.func @tensor_identity(%arg0: !torch.tensor) -> !torch.tensor {
  return %arg0 : !torch.tensor
}

// CHECK-LABEL: func.func @value_tensor_identity(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
// CHECK: return %[[TENSOR]] : !tvm_ffi.tensor
func.func @value_tensor_identity(%arg0: !torch.vtensor<[2,3],f32>)
    -> !torch.vtensor<[2,3],f32> {
  return %arg0 : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: func.func @string_identity(
// CHECK-SAME: %[[STRING:[a-zA-Z0-9_]+]]: !tvm_ffi.union<!tvm_ffi.raw_str, !tvm_ffi.small_str, !tvm_ffi.str>)
// CHECK-SAME: -> !tvm_ffi.union<!tvm_ffi.raw_str, !tvm_ffi.small_str, !tvm_ffi.str> {
// CHECK: return %[[STRING]] : !tvm_ffi.union<!tvm_ffi.raw_str, !tvm_ffi.small_str, !tvm_ffi.str>
func.func @string_identity(%arg0: !torch.str) -> !torch.str {
  return %arg0 : !torch.str
}

// Torch types without an external interface model preserve the Any fallback.
// CHECK-LABEL: func.func @unknown_identity(
// CHECK-SAME: %[[UNKNOWN:[a-zA-Z0-9_]+]]: !tvm_ffi.any) -> !tvm_ffi.any {
// CHECK: return %[[UNKNOWN]] : !tvm_ffi.any
func.func @unknown_identity(%arg0: !torch.number) -> !torch.number {
  return %arg0 : !torch.number
}
