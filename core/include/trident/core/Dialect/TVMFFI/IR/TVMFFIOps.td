//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIBase.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIInterfaces.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.td"

class TVMFFI_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVMFFI_Dialect, mnemonic, traits>;

class TVMFFI_TensorIndexedMetadataOp<string mnemonic>
    : TVMFFI_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyTVMFFITensorOrTorchType : $tensor, I64 : $index);
  let results = (outs I64 : $result);
  let assemblyFormat = "$tensor `[` $index `]` attr-dict `:` type($tensor)";
}

def TVMFFI_ArrayCreateOp
    : TVMFFI_Op<"array.create", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create a TVM FFI Array from semantic values";
  let arguments = (ins Variadic<AnyTVMFFISemanticType> : $elements);
  let results = (outs TVMFFI_ArrayType : $result);
}

def TVMFFI_ArrayGetItemOp : TVMFFI_Op<"array.get_item", [
    DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                              ["getObjectResultOwnership"]>]> {
  let summary = "Borrow an element from a TVM FFI Array";
  let arguments = (ins AnyTVMFFIArrayOrTorchType:$array,
                       TVMFFI_IntType:$index,
                       OptionalAttr<TypeAttr>:$element_type);
  let results = (outs AnyTypeOf<[AnyTVMFFISemanticType, AnyTorchType],
                                "TVM FFI semantic value or a Torch frontend value">
                 : $result);
  let assemblyFormat = "$array `[` $index `]` (`as` $element_type^)? attr-dict "
                       "`:` type($array) `,` type($index) `->` type($result)";
  let hasVerifier = 1;
}

def TVMFFI_ArrayLengthOp : TVMFFI_Op<"array.length", [Pure]> {
  let summary = "Read the length of a runtime TVM FFI array";
  let arguments = (ins AnyTVMFFIArrayOrTorchType : $array);
  let results = (outs I64 : $result);
  let assemblyFormat = "$array attr-dict `:` type($array)";
}

def TVMFFI_CallOp : TVMFFI_Op<"call", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Call a TVM FFI function defined in the current module";
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyTVMFFIAnyABIOrTorchType>:$operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyTVMFFIAnyABIOrTorchType> : $results);
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` "
                       "functional-type($operands, $results)";
}

def TVMFFI_CastOp
    : TVMFFI_Op<"cast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>,
        DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                                  ["getObjectResultOwnership"]>]> {
  let summary = "Widen a TVM FFI value into an any or union ABI type";
  // Torch types are accepted here as a transitional frontend boundary.  The
  // Torch-toTVMFFI conversion establishes the TVM FFI ABI representation before
  // this operation reaches LLVM lowering.
  let arguments = (ins AnyTVMFFIAnyABIOrTorchType : $value);
  let results = (outs AnyTVMFFIAnyABIType : $result);
  let assemblyFormat = "$value `:` type($value) `->` type($result) attr-dict";
}

def TVMFFI_ConstantOp
    : TVMFFI_Op<"constant", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create a TVM FFI scalar or owned tensor constant";
  let arguments = (ins AnyAttr : $value);
  let results = (outs AnyTVMFFISemanticType : $result);
  let hasVerifier = 1;
}

def TVMFFI_EqOp : TVMFFI_Op<"eq", [Pure]> {
  let summary = "Compare the ABI representation of two TVM FFI values";
  let arguments = (ins AnyTVMFFIAnyABIOrTorchType
                   : $lhs, AnyTVMFFIAnyABIOrTorchType
                   : $rhs);
  let results = (outs I1 : $result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

def TVMFFI_ExceptionOp
    : TVMFFI_Op<"exception", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create a Trident FFI exception object";
  let arguments = (ins StrAttr : $kind);
  let results = (outs TVMFFI_ExceptionType : $result);
  let assemblyFormat = "$kind attr-dict `:` type($result)";
}

def TVMFFI_FuncOp : TVMFFI_Op<"func", [
  AffineScope, AutomaticAllocationScope, FunctionOpInterface, IsolatedFromAbove,
  OpAsmOpInterface
]> {
  let summary = "A TVM FFI function";
  let description = [{
    Defines a function that can be called through the TVM FFI interface.

    Example:

    ```mlir
    tvm_ffi.func @add(%a: !tvm_ffi.int, %b: !tvm_ffi.int) -> !tvm_ffi.int {
      ...
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<UnitAttr>:$emit_tvm_ffi_abi,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion : $body);

  let builders = [OpBuilder<(ins
    "::llvm::StringRef":$name, "::mlir::FunctionType":$type,
    CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion();
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes();
    ::llvm::ArrayRef<::mlir::Type> getResultTypes();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
}

def TVMFFI_FunctionCallOp : TVMFFI_Op<"FunctionCall", [
    DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                              ["getObjectOperandOwnership"]>]> {
  let summary = "Call a TVM FFI function handle";
  let arguments = (ins TVMFFI_FunctionType:$callee,
                       Variadic<AnyTVMFFISemanticType>:$arguments);
  let results = (outs Variadic<AnyTVMFFISemanticType>:$results);
  let assemblyFormat = "$callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $results)";
}

def TVMFFI_FunctionGetGlobalOp
    : TVMFFI_Op<"FunctionGetGlobal", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Get a TVM FFI global function by name";
  let arguments = (ins StrAttr:$name);
  let results = (outs TVMFFI_FunctionType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

def TVMFFI_GetOp : TVMFFI_Op<"get", [Pure]> {
  let summary = "Extract a native boolean from a TVM FFI boolean";
  let arguments = (ins TVMFFI_BoolType:$operand);
  let results = (outs I1:$result);
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}

def TVMFFI_ObjectDecRefOp : TVMFFI_Op<"ObjectDecRef", [
    DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                              ["getObjectOperandOwnership"]>]> {
  let summary = "Decrement a TVM FFI object reference";
  let arguments = (ins AnyTVMFFIObjectOrAnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFI_ObjectIncRefOp : TVMFFI_Op<"ObjectIncRef", [
    DeclareOpInterfaceMethods<TVMFFI_ObjectOwnershipOpInterface,
                              ["getObjectOperandOwnership"]>]> {
  let summary = "Increment a TVM FFI object reference";
  let arguments = (ins AnyTVMFFIObjectOrAnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFI_ReturnOp
    : TVMFFI_Op<"return", [Pure, HasParent<"FuncOp">, ReturnLike, Terminator]> {
  let summary = "Return operation inside a TVM FFI function";
  let description = [{
    Returns values from a TVM FFI function. The operand count and types must
    match the function signature result types.

    Example:

    ```mlir
    tvm_ffi.return %result : i64
    ```
  }];

  // Torch types are accepted here as a transitional frontend boundary. The
  // Torch-to-TVMFFI pass rewrites them before lifetime-sensitive operations.
  let arguments = (ins Variadic<AnyTVMFFIAnyABIOrTorchType> : $operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def TVMFFI_TensorDeviceOp : TVMFFI_Op<"tensor.device", [Pure]> {
  let summary = "Read DLPack device fields from a runtime tensor";
  let arguments = (ins AnyTVMFFITensorOrTorchType:$tensor);
  let results = (outs I32:$device_type, I32:$device_index);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorLiteralOp
    : TVMFFI_Op<"tensor.literal", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create an owned TVM FFI tensor from a dense literal";
  let arguments = (ins ElementsAttr:$value);
  let results = (outs TVMFFI_TensorType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def TVMFFI_TensorCloneOp
    : TVMFFI_Op<"tensor.clone", [TVMFFI_ObjectOwnershipOpInterface]> {
  let summary = "Create an independent tensor copy preserving the layout";
  let arguments = (ins TVMFFI_TensorType:$tensor);
  let results = (outs TVMFFI_TensorType:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor) `->` type($result)";
}

def TVMFFI_TensorCopyOp
    : TVMFFI_Op<"tensor.copy_", []> {
  let summary = "Copy tensor contents into an existing tensor in place";
  let arguments = (ins TVMFFI_TensorType:$destination,
                       TVMFFI_TensorType:$source);
  let assemblyFormat = "$destination `,` $source attr-dict `:` type($destination)";
}

def TVMFFI_TensorDimOp : TVMFFI_Op<"tensor.dim", [Pure]> {
  let summary = "Read the rank of a runtime tensor";
  let arguments = (ins AnyTVMFFITensorOrTorchType:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorDTypeOp : TVMFFI_Op<"tensor.dtype", [Pure]> {
  let summary = "Read DLPack dtype fields from a runtime tensor";
  let arguments = (ins AnyTVMFFITensorOrTorchType:$tensor);
  let results = (outs I8:$code, I8:$bits, I16:$lanes);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorSizeOp
    : TVMFFI_TensorIndexedMetadataOp<"tensor.size"> {
  let summary = "Read a runtime tensor dimension size";
}

def TVMFFI_TensorStorageOffsetOp
    : TVMFFI_Op<"tensor.storage_offset", [Pure]> {
  let summary = "Read a runtime tensor storage offset";
  let arguments = (ins AnyTVMFFITensorOrTorchType:$tensor);
  let results = (outs I64:$result);
  let assemblyFormat = "$tensor attr-dict `:` type($tensor)";
}

def TVMFFI_TensorStrideOp
    : TVMFFI_TensorIndexedMetadataOp<"tensor.stride"> {
  let summary = "Read a runtime tensor stride";
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
