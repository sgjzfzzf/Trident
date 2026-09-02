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
// CHECK: %[[FUNC:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.runtime.tvm_ffi_to_torch_type" : !tvm_ffi.function
// CHECK: %[[TYPE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNC]](%[[DTYPE]]) : (!tvm_ffi.dtype) -> !tvm_ffi.int
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

// Scalar Torch types retain their corresponding semantic TVM FFI types.
// CHECK-LABEL: func.func @scalar_identities(
// CHECK-SAME: %[[BOOL:[a-zA-Z0-9_]+]]: !tvm_ffi.bool,
// CHECK-SAME: %[[FLOAT:[a-zA-Z0-9_]+]]: !tvm_ffi.float,
// CHECK-SAME: %[[INT:[a-zA-Z0-9_]+]]: !tvm_ffi.int,
// CHECK-SAME: %[[NONE:[a-zA-Z0-9_]+]]: !tvm_ffi.none)
// CHECK-SAME: -> (!tvm_ffi.bool, !tvm_ffi.float, !tvm_ffi.int, !tvm_ffi.none) {
// CHECK: return %[[BOOL]], %[[FLOAT]], %[[INT]], %[[NONE]] : !tvm_ffi.bool, !tvm_ffi.float, !tvm_ffi.int, !tvm_ffi.none
func.func @scalar_identities(
    %bool: !torch.bool,
    %float: !torch.float,
    %int: !torch.int,
    %none: !torch.none)
    -> (!torch.bool, !torch.float, !torch.int, !torch.none) {
  return %bool, %float, %int, %none
      : !torch.bool, !torch.float, !torch.int, !torch.none
}

// Tensor and value-tensor types share the semantic TVM FFI tensor type.
// CHECK-LABEL: func.func @tensor_identities(
// CHECK-SAME: %[[TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor,
// CHECK-SAME: %[[VALUE_TENSOR:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
// CHECK-SAME: -> (!tvm_ffi.tensor, !tvm_ffi.tensor) {
// CHECK: return %[[TENSOR]], %[[VALUE_TENSOR]] : !tvm_ffi.tensor, !tvm_ffi.tensor
func.func @tensor_identities(
    %tensor: !torch.tensor,
    %value_tensor: !torch.vtensor<[2,3],f32>)
    -> (!torch.tensor, !torch.vtensor<[2,3],f32>) {
  return %tensor, %value_tensor : !torch.tensor, !torch.vtensor<[2,3],f32>
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
