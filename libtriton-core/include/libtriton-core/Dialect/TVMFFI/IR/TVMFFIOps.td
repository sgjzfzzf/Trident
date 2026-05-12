#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "libtriton-core/Dialect/DLPack/IR/DLPackTypes.td"
include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIBase.td"
include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.td"

class TVMFFI_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVMFFI_Dialect, mnemonic, traits> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def TVMFFI_ToOp : TVMFFI_Op<"to", [MemoryEffects<[MemRead]>]> {
  let summary = "Convert values across the TVMFFI bridge type set";
  let description = [{
    Performs type -
        directed conversion between TVMFFI boundary types.Supported types
    are: integer types,
    f64,
    !llvm.ptr,
    !tvm_ffi.any,
    !tvm_ffi.object_handle,
    and !dlpack.tensor
        .The concrete lowering is selected from input and output types.
  }];
  let arguments = (ins AnyType : $input);
  let results = (outs AnyType : $output);
  let hasVerifier = 1;
}

def TVMFFI_AnyFromLLVMOp : TVMFFI_Op<"any_from_llvm", [Pure]> {
  let summary = "Convert an LLVM TVMFFIAny descriptor struct to !tvm_ffi.any";
  let description = [{Bridges an LLVM struct value carrying the TVMFFIAny ABI
                      into !tvm_ffi.any for high-level unboxing operations.
  }];
  let arguments = (ins LLVM_AnyStruct : $input);
  let results = (outs TVMFFI_AnyType : $output);
}

def TVMFFI_AnyToLLVMOp : TVMFFI_Op<"any_to_llvm", [Pure]> {
  let summary = "Convert !tvm_ffi.any to its LLVM TVMFFIAny descriptor struct";
  let description = [{Bridges !tvm_ffi.any to the LLVM struct value carrying
                      the TVMFFIAny ABI for low-level stores and calls.
  }];
  let arguments = (ins TVMFFI_AnyType : $input);
  let results = (outs LLVM_AnyStruct : $output);
}

def TVMFFI_TensorFromDLPackOp
    : TVMFFI_Op<"tensor_from_dlpack", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call TVMFFITensorFromDLPack and return object handle";
  let description = [{
    Models `TVMFFITensorFromDLPack(from, require_alignment, require_contiguous,
    out)` with `out` materialized by lowering. The op returns the produced
    TVMFFIObjectHandle directly.
  }];
  let arguments = (ins DLPack_DLManagedTensor
                   : $from, I32
                   : $require_alignment, I32
                   : $require_contiguous);
  let results = (outs TVMFFI_ObjectHandleType : $output);
  let assemblyFormat = "$from `,` $require_alignment `,` $require_contiguous "
                       "attr-dict `:` type($from) `,` type($require_alignment) "
                       "`,` type($require_contiguous) `->` type($output)";
}

def TVMFFI_ErrorSetRaisedFromCStrOp
    : Op<TVMFFI_Dialect, "error_set_raised_from_c_str",
         [MemoryEffects<[MemWrite]>]> {
  let summary = "Call TVMFFIErrorSetRaisedFromCStr";
  let description = [{Models `TVMFFIErrorSetRaisedFromCStr(kind, message)`.}];
  let arguments = (ins LLVM_AnyPointer : $kind, LLVM_AnyPointer : $message);
  let assemblyFormat = "$kind `,` $message attr-dict `:` type($kind) `,` "
                       "type($message)";
}

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
