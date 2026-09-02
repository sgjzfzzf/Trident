//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_TD_

include "mlir/IR/OpBase.td"

def TVMFFI_TypeIndexInterface
    : TypeInterface<"TVMFFITypeIndexInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  let description = [{
    Provides the concrete TVM FFI runtime type index for a semantic type.
  }];
  let methods = [
    StaticInterfaceMethod<
      "Returns the concrete TVM FFI runtime type index.",
      "int", "getTypeIndex", (ins)
    >
  ];
}

def TVMFFI_NativeTypeInterface
    : TypeInterface<"TVMFFINativeTypeInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  let description = [{
    Provides the native MLIR type used when a TVM FFI value is passed to a
    kernel.
  }];
  let methods = [
    InterfaceMethod<
      "Returns the corresponding native MLIR type.",
      "::mlir::Type", "getNativeType", (ins)
    >
  ];
}

def TVMFFI_ObjectOwnershipOpInterface
    : OpInterface<"ObjectOwnershipOpInterface"> {
  let cppNamespace = "::trident::tvm_ffi";
  let description = [{
    Describes how an operation produces and consumes object-typed TVM FFI
    values, ABI containers that may dynamically hold objects, and transitional
    Torch object values. Runtime reference operations inspect the type index of
    `!tvm_ffi.any` and `!tvm_ffi.union` values before changing a reference.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the ownership kind of the indexed result.",
      "::trident::tvm_ffi::ObjectResultOwnership",
      "getObjectResultOwnership", (ins "uint32_t":$index), [{}],
      /*defaultImplementation=*/[{
        return ::trident::tvm_ffi::ObjectResultOwnership::Owned;
      }]>,
    InterfaceMethod<
      "Returns the ownership effect of the indexed operand.",
      "::trident::tvm_ffi::ObjectOperandOwnership",
      "getObjectOperandOwnership", (ins "uint32_t":$index), [{}],
      /*defaultImplementation=*/[{
        return ::trident::tvm_ffi::ObjectOperandOwnership::Borrowed;
      }]>
  ];
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIINTERFACES_TD_
