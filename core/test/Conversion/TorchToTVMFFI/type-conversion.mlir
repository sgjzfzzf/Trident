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
