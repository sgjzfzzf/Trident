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
include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.td"

def TVMFFI_ABI : NativeTypeTrait<"TVMFFIABI">;
def TVMFFI_Object : NativeTypeTrait<"Object">;

class TVMFFI_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<TVMFFI_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class TVMFFIABI_Type<string name, string typeMnemonic, list<Trait> traits = [],
                      string baseCppClass =
                          "::trident::tvm_ffi::TVMFFIABIType">
    : TVMFFI_Type<name, typeMnemonic, !listconcat([TVMFFI_ABI], traits),
                  baseCppClass>;

class TVMFFIIndexed_Type<string name, string typeMnemonic, int typeIndex,
                         list<Trait> traits = [], string extraDeclarations = "",
                         string baseCppClass =
                             "::trident::tvm_ffi::TVMFFIABIType">
    : TVMFFIABI_Type<name, typeMnemonic,
                     !listconcat([TVMFFI_TypeIndexInterface], traits),
                     baseCppClass> {
  let extraClassDeclaration = [{
    static constexpr int getTypeIndex() { return }] # typeIndex # [{; }
  }] # extraDeclarations;
}

class TVMFFIObject_Type<string name, string typeMnemonic,
                        list<Trait> traits = [], string extraDeclarations = "">
    : TVMFFIABI_Type<name, typeMnemonic, !listconcat([TVMFFI_Object], traits),
                     "::trident::tvm_ffi::TVMFFIObjectType"> {
  let extraClassDeclaration = extraDeclarations;
}

class TVMFFIIndexedObject_Type<string name, string typeMnemonic, int typeIndex,
                               list<Trait> traits = [],
                               string extraDeclarations = "">
    : TVMFFIIndexed_Type<name, typeMnemonic, typeIndex,
                         !listconcat(
                             [TVMFFI_Object,
                              DeclareTypeInterfaceMethods<
                                  TVMFFI_NativeTypeInterface>],
                             traits),
                         extraDeclarations,
                         "::trident::tvm_ffi::TVMFFIObjectType">;

def TVMFFI_AnyType : TVMFFIABI_Type<"Any", "any">;
def TVMFFI_ArrayType
    : TVMFFIIndexedObject_Type<"Array", "array", 71>;
def TVMFFI_BoolType
    : TVMFFIIndexed_Type<"Bool", "bool", 2,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>]>;
def TVMFFI_DeviceType
    : TVMFFIIndexed_Type<"Device", "device", 6>;
def TVMFFI_DTypeType
    : TVMFFIIndexed_Type<"DType", "dtype", 5, [], [{
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, int64_t code,
                               int64_t bits, int64_t lanes = 1);
  }]>;
def TVMFFI_ExceptionType
    : TVMFFIObject_Type<"Exception", "exception",
                        [DeclareTypeInterfaceMethods<
                            TVMFFI_TypeIndexInterface>]>;
def TVMFFI_FloatType
    : TVMFFIIndexed_Type<"Float", "float", 3,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>]>;
def TVMFFI_FunctionType
    : TVMFFIIndexedObject_Type<"Function", "function", 68>;
def TVMFFI_IntType
    : TVMFFIIndexed_Type<"Int", "int", 1,
                         [DeclareTypeInterfaceMethods<
                             TVMFFI_NativeTypeInterface>], [{
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, int64_t value);
    static ::mlir::Value build(::mlir::OpBuilder &builder,
                               ::mlir::Location loc, ::mlir::Value value);
  }]>;
def TVMFFI_NoneType : TVMFFIIndexed_Type<"None", "none", 0>;
def TVMFFI_ObjectType
    : TVMFFIObject_Type<"Object", "object">;
def TVMFFI_RawStrType : TVMFFIIndexed_Type<"RawStr", "raw_str", 8>;
def TVMFFI_SmallStrType
    : TVMFFIIndexed_Type<"SmallStr", "small_str", 11>;
def TVMFFI_StrType
    : TVMFFIIndexedObject_Type<"Str", "str", 65>;
def TVMFFI_TensorType
    : TVMFFIIndexedObject_Type<"Tensor", "tensor", 70,
                               []>;
def TVMFFI_UnionType : TVMFFIABI_Type<"Union", "union"> {
  let summary = "A finite set of TVM FFI ABI alternatives";
  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$types);
  let assemblyFormat = "`<` $types `>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool contains(::mlir::Type type) const;
  }];
}

def AnyTVMFFIAnyABIType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_DeviceType,
  TVMFFI_DTypeType, TVMFFI_ExceptionType, TVMFFI_FloatType,
  TVMFFI_FunctionType, TVMFFI_IntType, TVMFFI_NoneType, TVMFFI_ObjectType,
  TVMFFI_RawStrType, TVMFFI_SmallStrType, TVMFFI_StrType, TVMFFI_TensorType,
  TVMFFI_UnionType
], "TVM FFI value with a TVMFFIAny ABI representation">;
def AnyTVMFFIGetResultType : AnyTypeOf<[
  AnyType, TVMFFI_ObjectType
], "TVM FFI get result">;
def AnyTVMFFIGetOperandType : AnyTypeOf<[
  TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_FloatType,
  TVMFFI_FunctionType, TVMFFI_IntType, TVMFFI_StrType, TVMFFI_TensorType
], "TVM FFI value with a native type">;
def AnyTVMFFIAnyABIOrTorchType : AnyTypeOf<[
  AnyTVMFFIAnyABIType, AnyTorchType, TorchExt_DTypeType
], "TVM FFI ABI value or a Torch frontend value">;
def AnyTVMFFIArrayOrTorchType : AnyTypeOf<[
  TVMFFI_ArrayType, AnyTorchListType, Torch_TupleType, Torch_AnyType
], "TVM FFI array or a Torch frontend container value">;
def AnyTVMFFIObjectOrAnyType : AnyTypeOf<[
  AnyTorchTensorType, TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_ExceptionType,
  TVMFFI_FunctionType, TVMFFI_StrType, TVMFFI_TensorType, TVMFFI_UnionType
], "TVM FFI object value or !tvm_ffi.any">;
def AnyTVMFFISemanticType : AnyTypeOf<[
  TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_DeviceType, TVMFFI_DTypeType,
  TVMFFI_FloatType,
  TVMFFI_IntType, TVMFFI_NoneType, TVMFFI_RawStrType, TVMFFI_SmallStrType,
  TVMFFI_StrType, TVMFFI_TensorType, TVMFFI_UnionType
], "TVM FFI semantic value">;
def AnyTVMFFITensorOrTorchType : AnyTypeOf<[
  AnyTorchTensorType, TVMFFI_TensorType
], "TVM FFI tensor or a Torch frontend tensor value">;

#endif
