//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTATTRS_TD_
#define TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTATTRS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "trident/core/Dialect/TorchExt/IR/TorchExtInterfaces.td"

class TorchExt_DTypeAttrDef<string attrName, string attrMnemonic,
                            string attrSummary>
    : AttrDef<TorchExt_Dialect, attrName, [
        DeclareAttrInterfaceMethods<TorchExt_DTypeAttr>
      ]> {
  let mnemonic = attrMnemonic;
  let summary = attrSummary;
}

def TorchExt_BFloat16Attr : TorchExt_DTypeAttrDef<
    "BFloat16", "bfloat16", "The torch.bfloat16 dtype">;
def TorchExt_BoolAttr : TorchExt_DTypeAttrDef<
    "Bool", "bool", "The torch.bool dtype">;
def TorchExt_Float4E2M1FNX2Attr : TorchExt_DTypeAttrDef<
    "Float4E2M1FNX2", "float4_e2m1fn_x2", "The torch.float4_e2m1fn_x2 dtype">;
def TorchExt_Float8E4M3FNAttr : TorchExt_DTypeAttrDef<
    "Float8E4M3FN", "float8_e4m3fn", "The torch.float8_e4m3fn dtype">;
def TorchExt_Float8E4M3FNUZAttr : TorchExt_DTypeAttrDef<
    "Float8E4M3FNUZ", "float8_e4m3fnuz", "The torch.float8_e4m3fnuz dtype">;
def TorchExt_Float8E5M2Attr : TorchExt_DTypeAttrDef<
    "Float8E5M2", "float8_e5m2", "The torch.float8_e5m2 dtype">;
def TorchExt_Float8E5M2FNUZAttr : TorchExt_DTypeAttrDef<
    "Float8E5M2FNUZ", "float8_e5m2fnuz", "The torch.float8_e5m2fnuz dtype">;
def TorchExt_Float8E8M0FNUAttr : TorchExt_DTypeAttrDef<
    "Float8E8M0FNU", "float8_e8m0fnu", "The torch.float8_e8m0fnu dtype">;
def TorchExt_Complex32Attr : TorchExt_DTypeAttrDef<
    "Complex32", "complex32", "The torch.complex32 dtype">;
def TorchExt_Complex64Attr : TorchExt_DTypeAttrDef<
    "Complex64", "complex64", "The torch.complex64 dtype">;
def TorchExt_Complex128Attr : TorchExt_DTypeAttrDef<
    "Complex128", "complex128", "The torch.complex128 dtype">;
def TorchExt_Float16Attr : TorchExt_DTypeAttrDef<
    "Float16", "float16", "The torch.float16 dtype">;
def TorchExt_Float32Attr : TorchExt_DTypeAttrDef<
    "Float32", "float32", "The torch.float32 dtype">;
def TorchExt_Float64Attr : TorchExt_DTypeAttrDef<
    "Float64", "float64", "The torch.float64 dtype">;
def TorchExt_Int8Attr : TorchExt_DTypeAttrDef<
    "Int8", "int8", "The torch.int8 dtype">;
def TorchExt_Int1Attr : TorchExt_DTypeAttrDef<
    "Int1", "int1", "The torch.int1 dtype">;
def TorchExt_Int2Attr : TorchExt_DTypeAttrDef<
    "Int2", "int2", "The torch.int2 dtype">;
def TorchExt_Int3Attr : TorchExt_DTypeAttrDef<
    "Int3", "int3", "The torch.int3 dtype">;
def TorchExt_Int4Attr : TorchExt_DTypeAttrDef<
    "Int4", "int4", "The torch.int4 dtype">;
def TorchExt_Int5Attr : TorchExt_DTypeAttrDef<
    "Int5", "int5", "The torch.int5 dtype">;
def TorchExt_Int6Attr : TorchExt_DTypeAttrDef<
    "Int6", "int6", "The torch.int6 dtype">;
def TorchExt_Int7Attr : TorchExt_DTypeAttrDef<
    "Int7", "int7", "The torch.int7 dtype">;
def TorchExt_Int16Attr : TorchExt_DTypeAttrDef<
    "Int16", "int16", "The torch.int16 dtype">;
def TorchExt_Int32Attr : TorchExt_DTypeAttrDef<
    "Int32", "int32", "The torch.int32 dtype">;
def TorchExt_Int64Attr : TorchExt_DTypeAttrDef<
    "Int64", "int64", "The torch.int64 dtype">;
def TorchExt_UInt8Attr : TorchExt_DTypeAttrDef<
    "UInt8", "uint8", "The torch.uint8 dtype">;
def TorchExt_UInt1Attr : TorchExt_DTypeAttrDef<
    "UInt1", "uint1", "The torch.uint1 dtype">;
def TorchExt_UInt2Attr : TorchExt_DTypeAttrDef<
    "UInt2", "uint2", "The torch.uint2 dtype">;
def TorchExt_UInt3Attr : TorchExt_DTypeAttrDef<
    "UInt3", "uint3", "The torch.uint3 dtype">;
def TorchExt_UInt4Attr : TorchExt_DTypeAttrDef<
    "UInt4", "uint4", "The torch.uint4 dtype">;
def TorchExt_UInt5Attr : TorchExt_DTypeAttrDef<
    "UInt5", "uint5", "The torch.uint5 dtype">;
def TorchExt_UInt6Attr : TorchExt_DTypeAttrDef<
    "UInt6", "uint6", "The torch.uint6 dtype">;
def TorchExt_UInt7Attr : TorchExt_DTypeAttrDef<
    "UInt7", "uint7", "The torch.uint7 dtype">;
def TorchExt_UInt16Attr : TorchExt_DTypeAttrDef<
    "UInt16", "uint16", "The torch.uint16 dtype">;
def TorchExt_UInt32Attr : TorchExt_DTypeAttrDef<
    "UInt32", "uint32", "The torch.uint32 dtype">;
def TorchExt_UInt64Attr : TorchExt_DTypeAttrDef<
    "UInt64", "uint64", "The torch.uint64 dtype">;

def TorchExt_SpecializationArrayAttr
    : TypedArrayAttrBase<TorchExt_SpecializationAttrInterface,
                         "array of Triton argument specializations">;

def TorchExt_ConstantSpecializationAttr
    : AttrDef<TorchExt_Dialect, "ConstantSpecialization", [
        DeclareAttrInterfaceMethods<TorchExt_SpecializationAttrInterface>
      ]> {
  let mnemonic = "constant_specialization";
  let summary = "Triton constant specialization assumption";
  let parameters = (ins "::mlir::Attribute":$value);
  let assemblyFormat = [{ `<` `value` `=` $value `>` }];
  let genVerifyDecl = 1;
}

def TorchExt_VariableSpecializationAttr
    : AttrDef<TorchExt_Dialect, "VariableSpecialization", [
        DeclareAttrInterfaceMethods<TorchExt_SpecializationAttrInterface>
      ]> {
  let mnemonic = "variable_specialization";
  let summary = "Triton variable specialization assumption";
  let parameters = (ins
    "::mlir::TypeAttr":$kind,
    DefaultValuedParameter<"uint64_t", "1">:$divisibility
  );
  let assemblyFormat = [{
    `<` `kind` `=` $kind (`,` `divisibility` `=` $divisibility^)? `>`
  }];
}

#endif // TRIDENT_CORE_DIALECT_TORCHEXT_IR_TORCHEXTATTRS_TD_
