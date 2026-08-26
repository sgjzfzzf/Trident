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
include "mlir/IR/CommonTypeConstraints.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"
include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIBase.td"

def TVMFFI_AnyABI : NativeTypeTrait<"AnyABI">;
def TVMFFI_Object : NativeTypeTrait<"Object">;

class TVMFFI_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TVMFFI_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class TVMFFIABI_Type<string name, string typeMnemonic, int typeIndex,
                      list<Trait> traits = []>
    : TVMFFI_Type<name, typeMnemonic, !listconcat([TVMFFI_AnyABI], traits)> {
  let extraClassDeclaration = [{
    static constexpr int getTypeIndex() { return }] # typeIndex # [{; }
  }];
}

def TVMFFI_AnyType : TVMFFIABI_Type<"Any", "any", -1>;
def TVMFFI_ArrayType
    : TVMFFIABI_Type<"Array", "array", 71,
                     [TVMFFI_Object]>;
def TVMFFI_BoolType
    : TVMFFIABI_Type<"Bool", "bool", 2>;
def TVMFFI_DeviceType
    : TVMFFIABI_Type<"Device", "device", 6>;
def TVMFFI_DTypeType
    : TVMFFIABI_Type<"DType", "dtype", 5>;
def TVMFFI_ExceptionType
    : TVMFFIABI_Type<"Exception", "exception", 67,
                     [TVMFFI_Object]>;
def TVMFFI_FloatType
    : TVMFFIABI_Type<"Float", "float", 3>;
// An owned handle returned by TVMFFIFunctionGetGlobal.
def TVMFFI_FunctionType
    : TVMFFIABI_Type<"Function", "function", 68,
                     [TVMFFI_Object]>;
def TVMFFI_IntType : TVMFFIABI_Type<"Int", "int", 1>;
def TVMFFI_NoneType : TVMFFIABI_Type<"None", "none", 0>;
def TVMFFI_StringType : TVMFFIABI_Type<"String", "string", 8>;
def TVMFFI_TensorType
    : TVMFFIABI_Type<"Tensor", "tensor", 70,
                     [TVMFFI_Object]>;

def AnyTVMFFIAnyABIType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_DeviceType,
  TVMFFI_DTypeType,
  TVMFFI_ExceptionType, TVMFFI_FloatType, TVMFFI_FunctionType, TVMFFI_IntType,
  TVMFFI_NoneType, TVMFFI_StringType, TVMFFI_TensorType
], "TVM FFI value with a TVMFFIAny ABI representation">;
def AnyTVMFFIAnyABIOrTorchType : AnyTypeOf<[
  AnyTVMFFIAnyABIType, AnyTorchType, TorchExt_DTypeType
], "TVM FFI ABI value or a Torch frontend value">;
def AnyTVMFFIArrayOrTorchType : AnyTypeOf<[
  TVMFFI_ArrayType, AnyTorchListType, Torch_TupleType, Torch_AnyType
], "TVM FFI array or a Torch frontend container value">;
def AnyTVMFFIObjectOrAnyType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_ExceptionType,
  TVMFFI_FunctionType, TVMFFI_TensorType, AnyTorchTensorType
], "TVM FFI object value or !tvm_ffi.any">;
def AnyTVMFFISemanticType : AnyTypeOf<[
  TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_DeviceType, TVMFFI_DTypeType,
  TVMFFI_FloatType,
  TVMFFI_IntType, TVMFFI_NoneType, TVMFFI_StringType, TVMFFI_TensorType,
  TorchExt_DTypeType
], "TVM FFI semantic value">;
def AnyTVMFFITensorOrTorchType : AnyTypeOf<[
  TVMFFI_TensorType, AnyTorchTensorType
], "TVM FFI tensor or a Torch frontend tensor value">;

#endif
