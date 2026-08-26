//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -generalize-aten-ops -convert-torch-to-tvm-ffi | FileCheck %s

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
// CHECK: %[[ABI_RESULT:[a-zA-Z0-9_]+]] = llvm.load %{{[a-zA-Z0-9_]+}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK: %[[TENSOR:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ABI_RESULT]] : !llvm.struct<(i32, i32, i64)> to !tvm_ffi.tensor
// CHECK: return %[[TENSOR]] : !tvm_ffi.tensor
func.func @literal_standalone() -> !torch.vtensor<[2,3],f32> {
  %literal = torch.vtensor.literal(
      dense<1.250000e+00> : tensor<2x3xf32>)
      : !torch.vtensor<[2,3],f32>
  return %literal : !torch.vtensor<[2,3],f32>
}
