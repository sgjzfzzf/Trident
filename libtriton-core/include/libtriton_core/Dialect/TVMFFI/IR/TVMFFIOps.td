#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "libtriton_core/Dialect/DLPack/IR/DLPackTypes.td"
include "libtriton_core/Dialect/TVMFFI/IR/TVMFFIBase.td"
include "libtriton_core/Dialect/TVMFFI/IR/TVMFFITypes.td"

class TVMFFI_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVMFFI_Dialect, mnemonic, traits> {
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def TVMFFI_FromIntOp : TVMFFI_Op<"from_int", [Pure]> {
  let summary = "Box i64 into TVMFFIAny";
  let description =
      [{Converts a 64 - bit integer input into the type -
        erased TVMFFIAny representation used at the runtime FFI boundary.}];
  let arguments = (ins I64 : $input);
  let results = (outs TVMFFI_AnyType : $output);
}

def TVMFFI_ToIntOp : TVMFFI_Op<"to_int", [Pure]> {
  let summary = "Unbox i64 from TVMFFIAny";
  let description = [{Converts a TVMFFIAny value into a 64 -
                      bit integer.This op is currently a conversion scaffold.}];
  let arguments = (ins TVMFFI_AnyType : $input);
  let results = (outs I64 : $output);
}

def TVMFFI_FromFloatOp : TVMFFI_Op<"from_float", [Pure]> {
  let summary = "Box f64 into TVMFFIAny";
  let description = [{
    Converts a 64-bit floating-point input into TVMFFIAny for FFI transport.
  }];
  let arguments = (ins F64 : $input);
  let results = (outs TVMFFI_AnyType : $output);
}

def TVMFFI_ToFloatOp : TVMFFI_Op<"to_float", [Pure]> {
  let summary = "Unbox f64 from TVMFFIAny";
  let description =
      [{Converts a TVMFFIAny value into a 64 - bit floating -
        point result.This op is currently a conversion scaffold.}];
  let arguments = (ins TVMFFI_AnyType : $input);
  let results = (outs F64 : $output);
}

def TVMFFI_FromStrOp : TVMFFI_Op<"from_str", [Pure]> {
  let summary = "Box string pointer into TVMFFIAny";
  let description =
      [{Converts an LLVM pointer value(typically used as an i8 * string pointer)
            into TVMFFIAny.}];
  let arguments = (ins LLVM_AnyPointer : $input);
  let results = (outs TVMFFI_AnyType : $output);
}

def TVMFFI_ToStrOp : TVMFFI_Op<"to_str", [Pure]> {
  let summary = "Unbox string pointer from TVMFFIAny";
  let description = [{Converts a TVMFFIAny value into an LLVM pointer result
                          intended to model a C -
                      style string pointer.}];
  let arguments = (ins TVMFFI_AnyType : $input);
  let results = (outs LLVM_AnyPointer : $output);
}

def TVMFFI_FromTensorOp : TVMFFI_Op<"from_tensor", [Pure]> {
  let summary = "Box DLTensor into TVMFFIAny";
  let description = [{Converts a DLTensor handle value into TVMFFIAny.}];
  let arguments = (ins DLPack_DLTensor : $input);
  let results = (outs TVMFFI_AnyType : $output);
}

def TVMFFI_ToTensorOp : TVMFFI_Op<"to_tensor", [Pure]> {
  let summary = "Unbox DLTensor from TVMFFIAny";
  let description = [{Converts a TVMFFIAny value into a DLPack DLTensor
                          handle.This op is currently a conversion scaffold.}];
  let arguments = (ins TVMFFI_AnyType : $input);
  let results = (outs DLPack_DLTensor : $output);
}

def TVMFFI_FromObjectOp : TVMFFI_Op<"from_object", [Pure]> {
  let summary = "Box object pointer into TVMFFIAny";
  let description =
      [{Converts an LLVM pointer value(typically modeling a void *object handle)
            into TVMFFIAny.}];
  let arguments = (ins LLVM_AnyPointer : $input);
  let results = (outs TVMFFI_AnyType : $output);
}

def TVMFFI_ToObjectOp : TVMFFI_Op<"to_object", [Pure]> {
  let summary = "Unbox object pointer from TVMFFIAny";
  let description = [{Converts a TVMFFIAny value into an LLVM pointer result
                          intended to model an opaque object pointer.}];
  let arguments = (ins TVMFFI_AnyType : $input);
  let results = (outs LLVM_AnyPointer : $output);
}

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
