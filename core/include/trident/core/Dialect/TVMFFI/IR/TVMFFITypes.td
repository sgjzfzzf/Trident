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

def TVMFFI_ABI : NativeTypeTrait<"TVMFFIABI">;
def TVMFFI_Object : NativeTypeTrait<"Object">;

class TVMFFI_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TVMFFI_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class TVMFFIABI_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TVMFFI_Type<name, typeMnemonic, !listconcat([TVMFFI_ABI], traits)>;

class TVMFFIIndexed_Type<string name, string typeMnemonic, int typeIndex,
                         list<Trait> traits = []>
    : TVMFFIABI_Type<name, typeMnemonic, traits> {
  let extraClassDeclaration = [{
    static constexpr int getTypeIndex() { return }] # typeIndex # [{; }
  }];
}

def TVMFFI_AnyType : TVMFFIIndexed_Type<"Any", "any", -1>;
def TVMFFI_ArrayType
    : TVMFFIIndexed_Type<"Array", "array", 71,
                         [TVMFFI_Object]>;
def TVMFFI_BoolType
    : TVMFFIIndexed_Type<"Bool", "bool", 2>;
def TVMFFI_DeviceType
    : TVMFFIIndexed_Type<"Device", "device", 6>;
def TVMFFI_DTypeType
    : TVMFFIIndexed_Type<"DType", "dtype", 5>;
def TVMFFI_ExceptionType
    : TVMFFIIndexed_Type<"Exception", "exception", 67,
                         [TVMFFI_Object]>;
def TVMFFI_FloatType
    : TVMFFIIndexed_Type<"Float", "float", 3>;
def TVMFFI_FunctionType
    : TVMFFIIndexed_Type<"Function", "function", 68,
                         [TVMFFI_Object]>;
def TVMFFI_IntType : TVMFFIIndexed_Type<"Int", "int", 1>;
def TVMFFI_NoneType : TVMFFIIndexed_Type<"None", "none", 0>;
def TVMFFI_RawStrType : TVMFFIIndexed_Type<"RawStr", "raw_str", 8>;
def TVMFFI_SmallStrType
    : TVMFFIIndexed_Type<"SmallStr", "small_str", 11>;
def TVMFFI_StrType
    : TVMFFIIndexed_Type<"Str", "str", 65,
                         [TVMFFI_Object]>;
def TVMFFI_TensorType
    : TVMFFIIndexed_Type<"Tensor", "tensor", 70,
                         [TVMFFI_Object]>;
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
  TVMFFI_DTypeType,
  TVMFFI_ExceptionType, TVMFFI_FloatType, TVMFFI_FunctionType, TVMFFI_IntType,
  TVMFFI_NoneType, TVMFFI_RawStrType, TVMFFI_SmallStrType, TVMFFI_StrType,
  TVMFFI_TensorType, TVMFFI_UnionType
], "TVM FFI value with a TVMFFIAny ABI representation">;
def AnyTVMFFIAnyABIOrTorchType : AnyTypeOf<[
  AnyTVMFFIAnyABIType, AnyTorchType, TorchExt_DTypeType
], "TVM FFI ABI value or a Torch frontend value">;
def AnyTVMFFIArrayOrTorchType : AnyTypeOf<[
  TVMFFI_ArrayType, AnyTorchListType, Torch_TupleType, Torch_AnyType
], "TVM FFI array or a Torch frontend container value">;
def AnyTVMFFIObjectOrAnyType : AnyTypeOf<[
  TVMFFI_AnyType, TVMFFI_ArrayType, TVMFFI_ExceptionType,
  TVMFFI_FunctionType, TVMFFI_StrType, TVMFFI_TensorType, TVMFFI_UnionType,
  AnyTorchTensorType
], "TVM FFI object value or !tvm_ffi.any">;
def AnyTVMFFISemanticType : AnyTypeOf<[
  TVMFFI_ArrayType, TVMFFI_BoolType, TVMFFI_DeviceType, TVMFFI_DTypeType,
  TVMFFI_FloatType,
  TVMFFI_IntType, TVMFFI_NoneType, TVMFFI_RawStrType, TVMFFI_SmallStrType,
  TVMFFI_StrType, TVMFFI_TensorType, TVMFFI_UnionType, TorchExt_DTypeType
], "TVM FFI semantic value">;
def AnyTVMFFISemanticOrTorchType : AnyTypeOf<[
  AnyTVMFFISemanticType, AnyTorchType
], "TVM FFI semantic value or a Torch frontend value">;
def AnyTVMFFITensorOrTorchType : AnyTypeOf<[
  TVMFFI_TensorType, AnyTorchTensorType
], "TVM FFI tensor or a Torch frontend tensor value">;

#endif
