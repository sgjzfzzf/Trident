//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// TVM FFI semantic types.
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFITYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIBase.td"

def TVMFFI_Object : NativeTypeTrait<"Object">;

class TVMFFI_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TVMFFI_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TVMFFI_ArrayType  : TVMFFI_Type<"Array", "array", [TVMFFI_Object]>;
def TVMFFI_BoolType   : TVMFFI_Type<"Bool", "bool">;
def TVMFFI_DeviceType : TVMFFI_Type<"Device", "device">;
def TVMFFI_FloatType  : TVMFFI_Type<"Float", "float">;
// An owned handle returned by TVMFFIFunctionGetGlobal.
def TVMFFI_FunctionType : TVMFFI_Type<"Function", "function", [TVMFFI_Object]>;
def TVMFFI_IntType    : TVMFFI_Type<"Int", "int">;
def TVMFFI_NoneType   : TVMFFI_Type<"None", "none">;

def TVMFFI_TensorType
    : TVMFFI_Type<"Tensor", "tensor", [TVMFFI_Object]>;

def AnyTVMFFISemanticType : Type<CPred<
  "::mlir::isa<::trident::tvm_ffi::BoolType, ::trident::tvm_ffi::IntType, "
  "::trident::tvm_ffi::FloatType, ::trident::tvm_ffi::NoneType, "
  "::trident::tvm_ffi::DeviceType, ::trident::tvm_ffi::ArrayType, "
  "::trident::tvm_ffi::TensorType>($_self)">,
  "TVM FFI semantic value">;

def AnyTVMFFIObjectType : Type<CPred<
  "$_self.hasTrait<::mlir::TypeTrait::Object>()">,
  "TVM FFI object value">;

#endif
