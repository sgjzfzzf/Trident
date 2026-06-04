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

def TVMFFI_GetTypeIndexOp : TVMFFI_Op<"get_type_index", [Pure]> {
  let summary = "Extract the TVMFFIAny type index as i32";
  let description = [{Returns the type tag stored in a !tvm_ffi.any value.}];
  let arguments = (ins TVMFFI_AnyType : $input);
  let results = (outs I32 : $output);
}

def TVMFFI_GetOpaquePtrOp
    : TVMFFI_Op<"get_opaque_ptr", [MemoryEffects<[MemRead]>]> {
  let summary = "Extract the cell pointer from a tvm_ffi.object_handle";
  // clang-format off
  let description = [{
    Given a !tvm_ffi.object_handle, computes a pointer to the cell data that
    follows the TVMFFIObject header by offsetting by sizeof(TVMFFIObject).
    The result is a !llvm.ptr.
    This is the MLIR equivalent of TVMFFITensorGetDLTensorPtr().
  }];
  // clang-format on
  let arguments = (ins TVMFFI_ObjectHandleType : $input);
  let results = (outs LLVM_AnyPointer : $output);
}

def TVMFFI_LoadOp : TVMFFI_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load a typed value from a !llvm.ptr";
  // clang-format off
  let description = [{
    Loads a typed value (e.g. !dlpack.tensor) from the memory pointed to by
    a !llvm.ptr. The output type determines the LLVM struct type used for the
    load. Currently supports !dlpack.tensor.
  }];
  // clang-format on
  let arguments = (ins LLVM_AnyPointer : $input);
  let results = (outs AnyType : $output);
}

def TVMFFI_StoreOp : Op<TVMFFI_Dialect, "store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a typed value as TVMFFIAny at the given pointer";
  // clang-format off
  let description = [{
    Stores the input value as a !tvm_ffi.any at the specified pointer.
    This is equivalent to to(input, !tvm_ffi.any) -> as(!llvm.struct) -> store,
    but as a single operation that avoids intermediate IR.
    Supported input types are those supported by tvm_ffi.to.
  }];
  // clang-format on
  let arguments = (ins AnyType : $value, LLVM_AnyPointer : $ptr);
  let results = (outs);
  let assemblyFormat =
      "$value `,` $ptr attr-dict `:` type($value) `,` type($ptr)";
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

def TVMFFI_FunctionGetGlobalOp : Op<TVMFFI_Dialect, "function_get_global",
                                    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Lookup a global TVM FFI function by name";
  // clang-format off
  let description = [{
    Looks up a global TVM FFI function registered in the system by name.
    Returns a handle to the function, or a null handle if not found.
    Models `TVMFFIFunctionGetGlobal(name, &out)`.
  }];
  // clang-format on
  let arguments = (ins StrAttr : $func_name);
  let results = (outs TVMFFI_ObjectHandleType : $output);
  let assemblyFormat = "$func_name attr-dict `:` type($output)";
}

def TVMFFI_FunctionCallOp : Op<TVMFFI_Dialect, "function_call",
                               [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Call a TVM FFI function";
  // clang-format off
  let description = [{
    Calls a TVM FFI function with the given arguments and returns the result.
    Models `TVMFFIFunctionCall(func, args, num_args, &result)`.
  }];
  // clang-format on
  let arguments = (ins
    TVMFFI_ObjectHandleType:$func,
    Variadic<TVMFFI_AnyType>:$args
  );
  let results = (outs TVMFFI_AnyType : $result);
  let assemblyFormat = "$func `(` $args `)` attr-dict `:` `(` type($args) `)` "
                       "`->` type($result)";
}

#endif // LIBTRITON_CORE_DIALECT_TVMFFI_IR_TVMFFIOPS_TD_
