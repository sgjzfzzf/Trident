#ifndef LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_
#define LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "libtriton-core/Dialect/DLPack/IR/DLPackBase.td"
include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.td"

class DLPack_Op<string mnemonic, list<Trait> traits = []>
    : Op<DLPack_Dialect, mnemonic, traits> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def DLPack_FromMemRefOwnedOp : DLPack_Op<"from_memref_owned", [Pure]> {
  let summary = "Convert an MLIR memref value to a DLManagedTensor handle";
  let description =
      [{Bridges a memref -
        typed MLIR value into a DLPack DLManagedTensor handle value
            .This op is intended as an interoperability scaffold and can be
                lowered during conversion to LLVM dialect.}];
  let arguments = (ins AnyMemRef : $input);
  let results = (outs DLPack_DLManagedTensor : $output);
}

def DLPack_FromMemRefBorrowedOp : DLPack_Op<"from_memref_borrowed", [Pure]> {
  let summary = "Convert an MLIR memref value to a borrowed DLTensor handle";
  let description = [{Bridges a memref -
        typed MLIR value into a non-owning DLPack DLTensor handle value
            .This op is intended for short-lived adapter paths and can be
                lowered during conversion to LLVM dialect.
  }];
  let arguments = (ins AnyMemRef : $input);
  let results = (outs DLPack_DLTensor : $output);
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

def DLPack_TensorFromLLVMOp : DLPack_Op<"tensor_from_llvm", [Pure]> {
  let summary = "Convert an LLVM DLTensor descriptor struct to !dlpack.tensor";
  let description = [{Bridges an LLVM struct value following the DLTensor ABI
                      into !dlpack.tensor for high-level tensor interop ops.
  }];
  let arguments = (ins LLVM_AnyStruct : $input);
  let results = (outs DLPack_DLTensor : $output);
}

def DLPack_TensorToLLVMOp : DLPack_Op<"tensor_to_llvm", [Pure]> {
  let summary = "Convert !dlpack.tensor to its LLVM descriptor struct";
  let description = [{
      Bridges !dlpack.tensor into the LLVM struct descriptor form used by low -
      level LLVM dialect ops.}];
  let arguments = (ins DLPack_DLTensor : $input);
  let results = (outs LLVM_AnyStruct : $output);
}

#endif // LIBTRITON_CORE_DIALECT_DLPACK_IR_DLPACKOPS_TD_