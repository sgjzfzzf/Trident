//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
#define TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFIBase.td"
include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.td"

class TVMFFI_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVMFFI_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Minimal value and lifetime operations.
//===----------------------------------------------------------------------===//

def TVMFFI_ArrayCreateOp : TVMFFI_Op<"array.create"> {
  let summary = "Create a TVM FFI Array from semantic values";
  let arguments = (ins Variadic<AnyTVMFFISemanticType>:$elements);
  let results = (outs TVMFFI_ArrayType:$result);
}

def TVMFFI_ArrayGetItemOp : TVMFFI_Op<"array.get_item"> {
  let summary = "Borrow an element from a TVM FFI Array";
  let arguments = (ins AnyTVMFFIObjectType:$array,
                       TVMFFI_IntType:$index,
                       OptionalAttr<TypeAttr>:$element_type);
  let results = (outs AnyTVMFFISemanticType:$result);
  let assemblyFormat = "$array `[` $index `]` (`as` $element_type^)? attr-dict `:` type($array) `,` type($index) `->` type($result)";
  let hasVerifier = 1;
}

def TVMFFI_CallOp : TVMFFI_Op<"call"> {
  let summary = "Call a TVM FFI function defined in the current module";
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$operands,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, $results)";
}

def TVMFFI_ConstantOp : TVMFFI_Op<"constant"> {
  let summary = "Create a TVM FFI scalar or owned tensor constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyTVMFFISemanticType:$result);
  let hasVerifier = 1;
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
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion : $body);

  let builders = [OpBuilder<(ins
    "::llvm::StringRef":$name, "::mlir::FunctionType":$type,
    CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs,
    CArg<"::mlir::ArrayAttr", "{}">:$argAttrs,
    CArg<"::mlir::ArrayAttr", "{}">:$resAttrs)
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

def TVMFFI_FunctionCallOp : TVMFFI_Op<"FunctionCall"> {
  let summary = "Call a TVM FFI function handle";
  let arguments = (ins TVMFFI_FunctionType:$callee,
                       Variadic<AnyTVMFFISemanticType>:$arguments);
  let results = (outs Variadic<AnyTVMFFISemanticType>:$results);
  let assemblyFormat = "$callee `(` $arguments `)` attr-dict `:` functional-type($arguments, $results)";
}

def TVMFFI_FunctionGetGlobalOp : TVMFFI_Op<"FunctionGetGlobal"> {
  let summary = "Get a TVM FFI global function by name";
  let arguments = (ins StrAttr:$name);
  let results = (outs TVMFFI_FunctionType:$result);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

def TVMFFI_ObjectDecRefOp : TVMFFI_Op<"ObjectDecRef"> {
  let summary = "Decrement a TVM FFI object reference";
  let arguments = (ins AnyType:$object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFI_ObjectIncRefOp : TVMFFI_Op<"ObjectIncRef"> {
  let summary = "Increment a TVM FFI object reference";
  let arguments = (ins AnyType:$object);
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
  let arguments = (ins Variadic<AnyType> : $operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
