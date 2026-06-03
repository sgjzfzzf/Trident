#ifndef LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
#define LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
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
  // clang-format off
  let description = [{
    Performs type-directed conversion between TVMFFI boundary types.
    Supported types are integer types, f64, !llvm.ptr, !tvm_ffi.any,
    !tvm_ffi.object_handle, and !dlpack.tensor. The concrete lowering is
    selected from input and output types.
  }];
  // clang-format on
  let arguments = (ins AnyType : $input);
  let results = (outs AnyType : $output);
  let hasVerifier = 1;
}

def TVMFFI_AsOp : TVMFFI_Op<"as", [Pure]> {
  let summary = "Convert between !tvm_ffi.any and its LLVM descriptor struct";
  // clang-format off
  let description = [{Bridges !tvm_ffi.any and the LLVM struct value carrying
                      the TVMFFIAny ABI in either direction.}];
  // clang-format on
  let arguments = (ins AnyTypeOf<[TVMFFI_AnyType, LLVM_AnyStruct]> : $input);
  let results = (outs AnyTypeOf<[TVMFFI_AnyType, LLVM_AnyStruct]> : $output);
  let hasVerifier = 1;
}

def TVMFFI_GetTypeIndexOp : TVMFFI_Op<"get_type_index", [Pure]> {
  let summary = "Extract the TVMFFIAny type index as i32";
  let description = [{Returns the type tag stored in a !tvm_ffi.any value.}];
  let arguments = (ins TVMFFI_AnyType : $input);
  let results = (outs I32 : $output);
}

def TVMFFI_EnvTensorAllocOp
    : TVMFFI_Op<"env_tensor_alloc", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Allocate tensor from TVMFFI environment allocator";
  // clang-format off
  let description =
      [{Builds a DLTensor prototype from the provided static shape and dtype,
        then allocates through `TVMFFIEnvTensorAlloc`. Device information is
        sourced from runtime lowering.}];
  // clang-format on
  let arguments = (ins TypeAttr : $dtype, DenseI64ArrayAttr : $shape);
  let results = (outs TVMFFI_ObjectHandleType : $output);
  let hasVerifier = 1;
  let assemblyFormat = "`dtype` `=` $dtype `,` `shape` `=` $shape attr-dict "
                       "`:` type($output)";
}

def TVMFFI_ObjectIncRefOp
    : Op<TVMFFI_Dialect, "object_inc_ref", [MemoryEffects<[MemWrite]>]> {
  let summary = "Increase refcount of a tvm_ffi.object_handle";
  let description = [{Models `TVMFFIObjectIncRef(obj)`.}];
  let arguments = (ins TVMFFI_ObjectHandleType : $object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
}

def TVMFFI_ObjectDecRefOp
    : Op<TVMFFI_Dialect, "object_dec_ref", [MemoryEffects<[MemWrite]>]> {
  let summary = "Decrease refcount of a tvm_ffi.object_handle";
  let description = [{Models `TVMFFIObjectDecRef(obj)`.}];
  let arguments = (ins TVMFFI_ObjectHandleType : $object);
  let assemblyFormat = "$object attr-dict `:` type($object)";
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
