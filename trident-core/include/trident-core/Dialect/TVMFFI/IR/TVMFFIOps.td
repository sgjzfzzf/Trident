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
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

include "trident-core/Dialect/TVMFFI/IR/TVMFFIBase.td"

class TVMFFI_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVMFFI_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def TVMFFI_FuncOp : TVMFFI_Op<"func", [
  AffineScope, AutomaticAllocationScope, FunctionOpInterface, IsolatedFromAbove,
  OpAsmOpInterface
]> {
  let summary = "A TVM FFI function";
  let description = [{
    Defines a function that can be called through the TVM FFI interface.

    Example:

    ```mlir
    tvm_ffi.func @add(%a: !torch.int, %b: !torch.int) -> !torch.int {
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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

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

let arguments = (ins Variadic<AnyTorchType> : $operands);

let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // TRIDENT_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
