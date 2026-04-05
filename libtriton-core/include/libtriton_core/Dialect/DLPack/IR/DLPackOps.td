#ifndef LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_
#define LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "libtriton_core/Dialect/DLPack/IR/DLPackBase.td"
include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.td"

class DLPack_Op<string mnemonic, list<Trait> traits = []>
    : Op<DLPack_Dialect, mnemonic, traits> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def DLPack_FromMemRefOp : DLPack_Op<"from_memref", [Pure]> {
  let summary = "Convert an MLIR memref value to a DLManagedTensor handle";
  let description =
      [{Bridges a memref -
        typed MLIR value into a DLPack DLManagedTensor handle value
            .This op is intended as an interoperability scaffold and can be
                lowered during conversion to LLVM dialect.}];
  let arguments = (ins AnyMemRef : $input);
  let results = (outs DLPack_DLManagedTensor : $output);
}

def DLPack_ViewOp : DLPack_Op<"view", [Pure]> {
  let summary = "Convert a DLManagedTensor handle to a plain DLTensor handle";
  let description =
      [{Extracts the DLTensor payload from a DLPack DLManagedTensor handle
            .This op is intended as an interoperability scaffold and can be
                lowered during conversion to LLVM dialect.}];
  let arguments = (ins DLPack_DLManagedTensor : $input);
  let results = (outs DLPack_DLTensor : $output);
}

def DLPack_ToMemRefOp : DLPack_Op<"to_memref", [Pure]> {
  let summary = "Convert a DLTensor handle to an MLIR memref value";
  let description =
      [{Bridges a DLPack DLTensor handle value to an MLIR memref -
        typed value
            .This op is intended as an interoperability scaffold and can be
                lowered during conversion to LLVM dialect.}];
  let arguments = (ins DLPack_DLTensor : $input);
  let results = (outs AnyMemRef : $output);
}

#endif // LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_