//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -convert-torchext-to-tvm-ffi | FileCheck %s

// The wrapper dtype is lowered to a TVM FFI dtype, while the conversion itself
// becomes a call to the runtime helper that produces the Torch scalar type.
// CHECK-LABEL: func.func @convert_dtype(
// CHECK-SAME: %[[DTYPE:.*]]: !torchext.dtype) {
// CHECK: %[[CONVERTED:.*]] = builtin.unrealized_conversion_cast %[[DTYPE]] : !torchext.dtype to !tvm_ffi.dtype
// CHECK: %[[FUNC:.*]] = tvm_ffi.FunctionGetGlobal "trident.runtime.tvm_ffi_to_torch_type"
// CHECK: %[[RESULT:.*]] = tvm_ffi.FunctionCall %[[FUNC]](%[[CONVERTED]])
// CHECK-SAME: -> !tvm_ffi.int
// CHECK: return
func.func @convert_dtype(%arg0: !torchext.dtype) {
  %0 = torchext.convert %arg0 : !torchext.dtype -> !torch.int
  return
}
